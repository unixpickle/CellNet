import Foundation
import HCBacktrace
import Honeycrisp

public class Muon: Optimizer {
  public var lr: Float
  public var momentum: Float
  public var weightDecay: Float
  public var nesterov: Bool
  public var nsSteps: Int

  public var moments: [String: Tensor] = [:]

  public init(
    _ parameters: [(String, Trainable.Parameter)],
    lr: Float,
    momentum: Float = 0.9,
    weightDecay: Float = 0.0,
    nesterov: Bool = true,
    nsSteps: Int = 5
  ) {
    self.lr = lr
    self.momentum = momentum
    self.weightDecay = weightDecay
    self.nesterov = nesterov
    self.nsSteps = nsSteps
    super.init(parameters)
  }

  @recordCaller static private func _newtonSchulz(_ g: Tensor, steps: Int) -> Tensor {
    #alwaysAssert(g.shape.count >= 2, "Muon can only be applied to 2D or higher Tensors")
    let (a, b, c) = (3.4445, -4.7750, 2.0315)
    let doTranspose = g.shape[g.shape.count - 2] > g.shape[g.shape.count - 1]
    var X = if doTranspose { g.swap(axis: -2, with: -1) } else { g }
    X =
      X
      / (X.flatten(startAxis: -2).pow(2).sum(axis: -1).sqrt().unsqueeze(axis: -1).unsqueeze(
        axis: -1
      ) + 1e-7)
    for _ in 0..<steps {
      let A = Tensor.matmul(a: X, transA: false, b: X, transB: true, transOut: false)
      let B = b * A + c * (A &* A)
      X = a * X + (B &* X)
    }

    if doTranspose { return X.swap(axis: -2, with: -1) } else { return X }
  }

  @recordCaller private func _step() {
    for (name, var param) in parameters {
      guard var g = param.grad else { continue }
      var v = moments[name] ?? Tensor(zerosLike: g)
      v = v * momentum + g * (1 - momentum)
      moments[name] = v
      g = if nesterov { g * (1 - momentum) + v * momentum } else { v }
      g = Muon.newtonSchulz(g, steps: nsSteps)
      if weightDecay != 0 { param.data = param.data! * (1 - lr * weightDecay) }
      let lrMult = pow(
        max(1.0, Float(g.shape[g.shape.count - 2]) / Float(g.shape[g.shape.count - 1])),
        0.5
      )
      param.data = param.data! - g * (lr * lrMult)
    }
  }

  /// An encodable object that contains all of the values that this optimizer
  /// tracks during optimization trajectories.
  public struct State: Codable, Sendable {
    public let moments: [String: TensorState]

    public init(moments: [String: TensorState] = [:]) { self.moments = moments }
  }

  public var state: TracedBlock<State> {
    let moments = moments
    return TracedBlock { State(moments: try await tensorsToStates(moments)) }
  }

  @recordCaller private func _loadState(_ state: State) throws {
    moments = statesToTensors(state.moments)
  }
}

private func tensorsToStates(_ d: [String: Tensor]) async throws -> [String: TensorState] {
  var result = [String: TensorState]()
  for (k, v) in d { result[k] = try await v.state() }
  return result
}

private func statesToTensors(_ d: [String: TensorState]) -> [String: Tensor] {
  var result = [String: Tensor]()
  for (k, v) in d { result[k] = Tensor(state: v) }
  return result
}
