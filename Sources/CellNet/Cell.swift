import ArgumentParser
import HCBacktrace
import Honeycrisp

public struct NetworkState: Sendable {
  public let inputs: Tensor
  public let targets: Tensor
  public let prevActivations: Tensor
  public let activations: Tensor
  public let cellStates: Tensor

  public init(
    inputs: Tensor,
    targets: Tensor,
    prevActivations: Tensor,
    activations: Tensor,
    cellStates: Tensor
  ) {
    self.inputs = inputs
    self.targets = targets
    self.prevActivations = prevActivations
    self.activations = activations
    self.cellStates = cellStates
  }

  public func with(
    inputs: Tensor? = nil,
    targets: Tensor? = nil,
    prevActivations: Tensor? = nil,
    activations: Tensor? = nil,
    cellStates: Tensor? = nil
  ) -> NetworkState {
    NetworkState(
      inputs: inputs ?? self.inputs,
      targets: targets ?? self.targets,
      prevActivations: prevActivations ?? self.prevActivations,
      activations: activations ?? self.activations,
      cellStates: cellStates ?? self.cellStates
    )
  }
}

public class Cell: Trainable {
  public enum Normalization: String, ExpressibleByArgument, CaseIterable, Sendable {
    case none
    case firstLayer
    case lastLayer
  }

  public let edgeCount: Int
  public let stateCount: Int
  public let normalization: Normalization

  @Child public var stateProj: Linear
  @Child public var edgeProj: Linear
  @Child public var prevEdgeProj: Linear
  @Child public var inOutProj: Linear
  @Child public var layer2: Linear
  @Child public var layer3: Linear

  public init(
    edgeCount: Int,
    stateCount: Int,
    hiddenSize: Int,
    normalization: Normalization = .none
  ) {
    self.edgeCount = edgeCount
    self.stateCount = stateCount
    self.normalization = normalization
    super.init()
    self.stateProj = Linear(inCount: stateCount, outCount: hiddenSize)
    self.edgeProj = Linear(inCount: edgeCount, outCount: hiddenSize)
    self.prevEdgeProj = Linear(inCount: edgeCount, outCount: hiddenSize)
    self.inOutProj = Linear(inCount: 2, outCount: hiddenSize)
    self.layer2 = Linear(inCount: hiddenSize, outCount: hiddenSize)
    self.layer3 = Linear(inCount: hiddenSize, outCount: stateCount * 2 + edgeCount * 2 + 1)
  }

  @recordCaller private func _callAsFunction(_ s: NetworkState) -> (
    outputs: Tensor, newActs: Tensor, newCellStates: Tensor
  ) {
    let inOut = Tensor(
      concat: [addInnerDimension(s.inputs, size: 1), addInnerDimension(s.targets, size: 1)],
      axis: -1
    )
    var h =
      stateProj(addInnerDimension(s.cellStates, size: stateCount))
      + prevEdgeProj(addInnerDimension(s.prevActivations, size: edgeCount))
      + edgeProj(addInnerDimension(s.activations, size: edgeCount)) + inOutProj(inOut)
    if normalization == .firstLayer {
      h = h.flatten(startAxis: 1).normalize(axis: -1, eps: 1e-5).reshape(h.shape)
    }
    h = h.gelu()
    h = self.layer2(h)
    h = h.gelu()
    if normalization == .lastLayer {
      h = h.flatten(startAxis: 1).normalize(axis: -1, eps: 1e-5).reshape(h.shape)
    }
    h = self.layer3(h)

    let rememberBias = 4 * Tensor(data: (0..<stateCount), dtype: .float32) / stateCount - 2

    let stateMask = (rememberBias + h[FullRange(count: h.shape.count - 1), 0..<stateCount]).flatten(
      startAxis: -2
    )
    let stateUpdate = h[FullRange(count: h.shape.count - 1), stateCount..<(stateCount * 2)].flatten(
      startAxis: -2
    )
    let outputs = h[FullRange(count: h.shape.count - 1), (stateCount * 2)]
    let actMask = h[
      FullRange(count: h.shape.count - 1),
      (stateCount * 2 + 1)..<(stateCount * 2 + edgeCount + 1)
    ].flatten(startAxis: -2)
    let actUpdate = h[FullRange(count: h.shape.count - 1), (stateCount * 2 + edgeCount + 1)...]
      .flatten(startAxis: -2)

    return (
      outputs: outputs,
      newActs: actMask.sigmoid() * s.prevActivations + (-actMask).sigmoid() * actUpdate,
      newCellStates: stateMask.sigmoid() * s.cellStates + (-stateMask).sigmoid() * stateUpdate
    )
  }
}

func addInnerDimension(_ t: Tensor, size: Int) -> Tensor {
  t.reshape(t.shape[..<(t.shape.count - 1)] + [t.shape.last! / size, size])
}
