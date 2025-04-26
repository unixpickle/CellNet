import ArgumentParser
import Foundation
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

  public func withNoise(std: Float) -> NetworkState {
    let prevActStd = prevActivations.variance(axis: 1, keepdims: true).sqrt() * std
    let actStd = activations.variance(axis: 1, keepdims: true).sqrt() * std
    let cellStateStd = cellStates.variance(axis: 1, keepdims: true).sqrt() * std
    return with(
      prevActivations: prevActivations + Tensor(randnLike: prevActivations) * prevActStd,
      activations: activations + Tensor(randnLike: activations) * actStd,
      cellStates: cellStates + Tensor(randnLike: cellStates) * cellStateStd
    )
  }
}

public class MetaLinear: Trainable {
  public enum Initialization: String, ExpressibleByArgument, CaseIterable, Sendable {
    case xavier
    case ortho
  }

  @Param(name: "weight") public var weight: Tensor

  public let castParams: Tensor.DType?

  public init(
    inCount: Int,
    outCount: Int,
    dtype: Tensor.DType = .float32,
    castParams: Tensor.DType? = nil,
    initialization: Initialization = .xavier
  ) {
    self.castParams = castParams
    super.init()
    switch initialization {
    case .xavier:
      self.weight =
        (Tensor(rand: [inCount, outCount], dtype: dtype) - 0.5)
        * (sqrt(3.0) / 0.5 / sqrt(Float(inCount)))
    case .ortho:
      let noise = Tensor(randn: [inCount, outCount], dtype: dtype)
      let (u, _, vt) = noise.svd(full: false)
      self.weight = u &* vt
    }
  }

  @recordCaller private func _callAsFunction(
    _ x: Tensor,
    weight: Tensor? = nil,
    weightGradBackend: Backend? = nil
  ) -> Tensor {
    if x.shape.count > 2 {
      let squashedBatch = x.reshape([
        x.shape[..<(x.shape.count - 1)].reduce(1, *), x.shape[x.shape.count - 1],
      ])
      let out = self(squashedBatch, weight: weight, weightGradBackend: weightGradBackend)
      return out.reshape(x.shape[..<(x.shape.count - 1)] + [out.shape[out.shape.count - 1]])
    }

    var weight = (weight ?? self.weight)
    weight = weight.cast(castParams ?? weight.dtype)

    if weight.shape.count == 3 {
      return Tensor.batchedMatmul(
        a: x.reshape([weight.shape[0], x.shape[0] / weight.shape[0], x.shape[1]]),
        transA: false,
        b: weight,
        transB: false,
        transOut: false
      ).flatten(endAxis: 1)
    } else {
      return Tensor.matmul(
        a: x,
        transA: false,
        b: weight,
        transB: false,
        transOut: false,
        bGradBackend: weightGradBackend
      )
    }
  }
}

public class Cell: Trainable {
  public enum Normalization: String, ExpressibleByArgument, CaseIterable, Sendable {
    case none
    case firstLayer
    case lastLayer
    case lastLayerDimwise
  }

  public let edgeCount: Int
  public let stateCount: Int
  public let normalization: Normalization
  public let actScale: Float

  @Child(name: "stateProj") public var stateProj: MetaLinear
  @Child(name: "edgeProj") public var edgeProj: MetaLinear
  @Child(name: "prevEdgeProj") public var prevEdgeProj: MetaLinear
  @Child(name: "inOutProj") public var inOutProj: MetaLinear
  @Child(name: "layer2") public var layer2: MetaLinear
  @Child(name: "layer3") public var layer3: MetaLinear

  public init(
    edgeCount: Int,
    stateCount: Int,
    hiddenSize: Int,
    normalization: Normalization = .none,
    initialization: MetaLinear.Initialization = .xavier,
    actScale: Float = 1.0
  ) {
    self.edgeCount = edgeCount
    self.stateCount = stateCount
    self.normalization = normalization
    self.actScale = actScale
    super.init()
    self.stateProj = MetaLinear(
      inCount: stateCount,
      outCount: hiddenSize,
      initialization: initialization
    )
    self.edgeProj = MetaLinear(
      inCount: edgeCount,
      outCount: hiddenSize,
      initialization: initialization
    )
    self.prevEdgeProj = MetaLinear(
      inCount: edgeCount,
      outCount: hiddenSize,
      initialization: initialization
    )
    self.inOutProj = MetaLinear(inCount: 2, outCount: hiddenSize, initialization: initialization)
    self.layer2 = MetaLinear(
      inCount: hiddenSize,
      outCount: hiddenSize,
      initialization: initialization
    )
    self.layer3 = MetaLinear(
      inCount: hiddenSize,
      outCount: stateCount * 2 + edgeCount * 2 + 1,
      initialization: initialization
    )
  }

  @recordCaller private func _callAsFunction(_ s: NetworkState, params: [String: Tensor]? = nil)
    -> (outputs: Tensor, newActs: Tensor, newCellStates: Tensor)
  {
    let inOut = Tensor(
      concat: [addInnerDimension(s.inputs, size: 1), addInnerDimension(s.targets, size: 1)],
      axis: -1
    )
    var h =
      stateProj(
        addInnerDimension(s.cellStates, size: stateCount),
        weight: params?["stateProj.weight"]
      )
      + prevEdgeProj(
        addInnerDimension(s.prevActivations, size: edgeCount),
        weight: params?["prevEdgeProj.weight"]
      )
      + edgeProj(
        addInnerDimension(s.activations, size: edgeCount),
        weight: params?["edgeProj.weight"]
      ) + inOutProj(inOut, weight: params?["inOutProj.weight"])
    if normalization == .firstLayer {
      h = h.flatten(startAxis: 1).normalize(axis: -1, eps: 1e-5).reshape(h.shape)
    }
    h = h.gelu()
    h = self.layer2(h, weight: params?["layer2.weight"])
    if actScale != 1 { h = h * actScale }
    h = h.gelu()
    if normalization == .lastLayer {
      h = h.flatten(startAxis: 1).normalize(axis: -1, eps: 1e-5).reshape(h.shape)
    } else if normalization == .lastLayerDimwise {
      h = h.normalize(axis: 1, eps: 1e-5)
    }
    h = self.layer3(h, weight: params?["layer3.weight"])

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
      newActs: actMask.sigmoid() * s.prevActivations + (-actMask).sigmoid() * actUpdate.tanh(),
      newCellStates: stateMask.sigmoid() * s.cellStates + (-stateMask).sigmoid()
        * stateUpdate.tanh()
    )
  }
}

func addInnerDimension(_ t: Tensor, size: Int) -> Tensor {
  t.reshape(t.shape[..<(t.shape.count - 1)] + [t.shape.last! / size, size])
}
