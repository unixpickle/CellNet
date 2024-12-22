import HCBacktrace
import Honeycrisp

public struct NetworkState: Sendable {
  public let inputs: Tensor
  public let targets: Tensor
  public let activations: Tensor
  public let cellStates: Tensor

  public init(inputs: Tensor, targets: Tensor, activations: Tensor, cellStates: Tensor) {
    self.inputs = inputs
    self.targets = targets
    self.activations = activations
    self.cellStates = cellStates
  }

  public func with(
    inputs: Tensor? = nil,
    targets: Tensor? = nil,
    activations: Tensor? = nil,
    cellStates: Tensor? = nil
  ) -> NetworkState {
    NetworkState(
      inputs: inputs ?? self.inputs,
      targets: targets ?? self.targets,
      activations: activations ?? self.activations,
      cellStates: cellStates ?? self.cellStates
    )
  }
}

public class Cell: Trainable {
  public let edgeCount: Int
  public let stateCount: Int

  @Child public var stateProj: Linear
  @Child public var edgeProj: Linear
  @Child public var inOutProj: Linear
  @Child public var layer2: Linear
  @Child public var layer3: Linear

  public init(edgeCount: Int, stateCount: Int, hiddenSize: Int) {
    self.edgeCount = edgeCount
    self.stateCount = stateCount
    super.init()
    self.stateProj = Linear(inCount: stateCount, outCount: hiddenSize)
    self.edgeProj = Linear(inCount: edgeCount, outCount: hiddenSize)
    self.inOutProj = Linear(inCount: 2, outCount: hiddenSize)
    self.layer2 = Linear(inCount: hiddenSize, outCount: hiddenSize)
    self.layer3 = Linear(inCount: hiddenSize, outCount: stateCount * 2 + edgeCount + 1)
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
      + edgeProj(addInnerDimension(s.activations, size: edgeCount)) + inOutProj(inOut)
    h = h.gelu()
    h = self.layer2(h)
    h = h.gelu()
    h = self.layer3(h)

    let rememberBias = 4 * Tensor(data: (0..<stateCount), dtype: .float32) / stateCount - 2

    let stateMask = (rememberBias + h[FullRange(count: h.shape.count - 1), 0..<stateCount]).flatten(
      startAxis: -2
    )
    let stateUpdate = h[FullRange(count: h.shape.count - 1), stateCount..<(stateCount * 2)].flatten(
      startAxis: -2
    )
    let outputs = h[FullRange(count: h.shape.count - 1), (stateCount * 2)]
    let newActs = h[FullRange(count: h.shape.count - 1), (stateCount * 2 + 1)...].flatten(
      startAxis: -2
    )

    return (
      outputs: outputs, newActs: newActs,
      newCellStates: stateMask.sigmoid() * s.cellStates + (-stateMask).sigmoid() * stateUpdate
    )
  }
}

func addInnerDimension(_ t: Tensor, size: Int) -> Tensor {
  t.reshape(t.shape[..<(t.shape.count - 1)] + [t.shape.last! / size, size])
}
