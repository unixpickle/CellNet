import HCBacktrace
import Honeycrisp

public struct NetworkState: Sendable {
  public let activations: Tensor
  public let cellStates: Tensor

  public init(activations: Tensor, cellStates: Tensor) {
    self.activations = activations
    self.cellStates = cellStates
  }

  public func with(activations: Tensor) -> NetworkState {
    NetworkState(activations: activations, cellStates: cellStates)
  }
}

public class Cell: Trainable {
  public let edgeCount: Int
  public let stateCount: Int

  @Child public var layer1: Linear
  @Child public var norm: LayerNorm
  @Child public var layer2: Linear
  @Child public var layer3: Linear

  public init(edgeCount: Int, stateCount: Int, hiddenSize: Int) {
    self.edgeCount = edgeCount
    self.stateCount = stateCount
    super.init()
    self.layer1 = Linear(inCount: edgeCount + stateCount, outCount: hiddenSize)
    self.layer2 = Linear(inCount: hiddenSize, outCount: hiddenSize)
    self.layer3 = Linear(inCount: hiddenSize, outCount: stateCount * 2 + edgeCount)
    self.layer3.bias = Tensor(randnLike: self.layer3.bias!) * 0.01
  }

  @recordCaller private func _callAsFunction(_ s: NetworkState) -> NetworkState {
    var h = Tensor(concat: [s.activations, s.cellStates], axis: -1)
    h = h / h.pow(2).mean(axis: -1, keepdims: true).clamp(min: 1e-5).sqrt()
    let featureSize = edgeCount + stateCount
    h = h.reshape(h.shape[..<(h.shape.count - 1)] + [h.shape.last! / featureSize, featureSize])
    h = self.layer1(h)
    h = h.gelu()
    h = self.layer2(h)
    h = h.gelu()
    h = self.layer3(h)

    let stateMask = h[FullRange(count: h.shape.count - 1), 0..<stateCount].flatten(startAxis: -2)
    let stateUpdate = h[FullRange(count: h.shape.count - 1), stateCount..<(stateCount * 2)].flatten(
      startAxis: -2
    )
    let newActs = h[FullRange(count: h.shape.count - 1), (stateCount * 2)...].flatten(startAxis: -2)

    return NetworkState(
      activations: newActs,
      cellStates: stateMask.sigmoid() * s.cellStates + (-stateMask).sigmoid() * stateUpdate
    )
  }
}
