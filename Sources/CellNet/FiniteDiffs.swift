import HCBacktrace
import Honeycrisp

/// Importance-sampled finite differences for gradient estimation.
public class FiniteDiffs {

  private struct History {
    public var history: [Float] = []

    mutating func append(_ x: Float, maxCount: Int) {
      history.append(x)
      if history.count > maxCount { history.remove(at: 0) }
    }

    func mean(placeholder: Float) -> Float {
      if history.isEmpty { return placeholder }
      return history.reduce(0.0, +) / Float(history.count)
    }
  }

  public struct State: Codable, Sendable {
    public let history: [[Float]]

    public init(history: [[Float]]) {
      self.history = history
    }
  }

  public struct AxisSample {
    public let axes: Tensor
    public let weights: Tensor
  }

  private let axisCount: Int
  private var axisHistory: [History]

  public let model: TrainableProto
  public let eps: Float
  public let evalCount: Int
  public let historyCount: Int
  public let randomProb: Float

  public init(
    model: TrainableProto,
    eps: Float,
    evalCount: Int,
    historyCount: Int = 2,
    randomProb: Float = 0.5
  ) {
    axisCount = model.parameters.map { $0.1.data?.shape.reduce(1, *) ?? 0 }.reduce(0, +)
    axisHistory = (0..<axisCount).map { _ in History() }

    self.model = model
    self.eps = eps
    self.evalCount = min(axisCount, evalCount)
    self.historyCount = historyCount
    self.randomProb = randomProb
  }

  @recordCaller private func _sampleAxes() -> AxisSample {
    let randomProbPerAxis = randomProb / Float(axisCount)
    let weights: [Float] = axisHistory.map { $0.mean(placeholder: 10000.0) }
    var probs = Tensor(data: weights)
    probs = probs / probs.sum()
    probs = probs * (1 - randomProb) + randomProbPerAxis
    let logits = probs.log()
    let bigBatch = logits.unsqueeze(axis: 0).repeating(axis: 0, count: evalCount)
    let gumbels = -(-Tensor(randLike: bigBatch).clamp(min: 1e-5).log()).log()
    let indices = (bigBatch + gumbels).argmax(axis: 1)
    let importanceWeights = 1 / probs.gather(axis: 0, indices: indices)
    return AxisSample(axes: indices, weights: importanceWeights / Float(evalCount))
  }

  @recordCaller private func _forwardBackward(
    sample: AxisSample,
    metric: Metrics.Key = .loss,
    batchSize: Int,
    lossFn: (Int, [String: Tensor]?) -> Metrics
  ) async throws -> Metrics {
    let evalCount = sample.axes.shape[0]
    return try await Tensor.withGrad(enabled: false) {
      var allMetrics = [Metrics]()

      for i in stride(from: 0, to: evalCount, by: batchSize) {
        let axes = sample.axes[i..<min(evalCount, i + batchSize)]
        let weights = sample.weights[i..<min(evalCount, i + batchSize)]
        let mbSize = axes.shape[0]
        let oneDirections = Tensor(constant: eps, shape: [mbSize, 1]).scatter(
          axis: 1,
          count: axisCount,
          indices: axes.unsqueeze(axis: 1)
        )
        let bothDirections = Tensor(stack: [oneDirections, -oneDirections], axis: 1).flatten(
          endAxis: 1
        )
        var params = [String: Tensor]()
        var offset = 0
        for (name, center) in model.parameters {
          guard let data = center.data else { continue }
          let size = data.shape.reduce(1, *)
          params[name] =
            data
            + bothDirections[..., offset..<(offset + size)].reshape(
              Array([mbSize * 2] + data.shape)
            )
          offset += size
        }
        let combinedMetrics = lossFn(mbSize * 2, params)
        #alwaysAssert(
          combinedMetrics.keys.contains(metric),
          "metric \(metric) not in returned metrics: \(combinedMetrics.keys)"
        )
        allMetrics.append(Metrics(dictionary: combinedMetrics.values.mapValues { $0.mean() }))

        let metricValues = combinedMetrics[metric]!.reshape([mbSize, 2])
        let deltas = (metricValues[..., 0] - metricValues[..., 1]) / (2 * eps)
        let weightedGrad = ((weights * deltas).unsqueeze(axis: 0) &* (oneDirections / eps)).squeeze(axis: 0)

        offset = 0
        for (_, var param) in model.parameters {
          guard let data = param.data else { continue }
          let size = data.shape.reduce(1, *)
          let subGrad = weightedGrad[offset..<(offset + size)].reshape(data.shape)
          if let g = param.grad { param.grad = g + subGrad } else { param.grad = subGrad }
          offset += size
        }

        for (axis, delta) in zip(try await axes.ints(), try await deltas.floats()) {
          axisHistory[axis].append(abs(delta), maxCount: historyCount)
        }
      }

      return Metrics.mean(allMetrics)
    }
  }

  public func state() -> State {
    State(history: axisHistory.map { $0.history })
  }

  public func loadState(_ state: State) {
    for (i, s) in state.history.enumerated() {
      axisHistory[i].history = s
    }
  }

}
