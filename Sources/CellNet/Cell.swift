import HCBacktrace
import Honeycrisp

public struct NetworkState {
  public let activations: Tensor
  public let states: Tensor

  public init(activations: Tensor, states: Tensor) {
    self.activations = activations
    self.states = states
  }
}

public class Cell: Trainable {
  public let edgeCount: Int
  public let stateCount: Int

  @Child public var layer1: Linear
  @Child public var layer2: Linear
  @Child public var layer3: Linear

  public init(edgeCount: Int, stateCount: Int, hiddenSize: Int) {
    self.edgeCount = edgeCount
    self.stateCount = stateCount
    super.init()
    self.layer1 = Linear(inCount: edgeCount + stateCount, outCount: hiddenSize)
    self.layer2 = Linear(inCount: hiddenSize, outCount: hiddenSize)
    self.layer2 = Linear(inCount: hiddenSize, outCount: stateCount * 2 + edgeCount)
  }

  @recordCaller private func _callAsFunction(_ s: NetworkState) -> NetworkState {
    var h = Tensor(concat: [s.activations, s.states], axis: -1)
    h = self.layer1(h)
    h = h.gelu()
    h = self.layer2(h)
    h = h.gelu()
    h = self.layer3(h)

    let stateMask = h[FullRange(count: h.shape.count - 1), 0..<stateCount]
    let stateUpdate = h[FullRange(count: h.shape.count - 1), stateCount..<(stateCount * 2)]
    let newActs = h[FullRange(count: h.shape.count - 1), (stateCount * 2)...]
    return NetworkState(
      activations: newActs,
      states: stateMask.sigmoid() * s.states + (-stateMask).sigmoid() * stateUpdate
    )
  }
}