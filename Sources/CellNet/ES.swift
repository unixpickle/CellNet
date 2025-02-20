import HCBacktrace
import Honeycrisp

/// Evolution strategies implementation.
public class ES {
  public let model: TrainableProto
  public let eps: Float
  public let directionCount: Int

  public init(model: TrainableProto, eps: Float, directionCount: Int) {
    self.model = model
    self.eps = eps
    self.directionCount = directionCount
  }

  @recordCaller private func _sampleNoises() -> [[Tensor?]] {
    Tensor.withGrad(enabled: false) {
      return (0..<directionCount).map { _ in
        model.parameters.map { if let data = $0.1.data { Tensor(randnLike: data) } else { nil } }
      }
    }
  }

  @recordCaller private func _forwardBackward(
    noises: [[Tensor?]],
    metric: Metrics.Key = .loss,
    batchSize: Int,
    lossFn: (Int, [String: Tensor]?) -> Metrics
  ) async throws -> Metrics {
    try await Tensor.withGrad(enabled: false) {
      var allMetrics = [Metrics]()

      for i in stride(from: 0, to: noises.count, by: batchSize) {
        let subNoises = noises[i..<min(noises.count, i + batchSize)]
        var params = [String: Tensor]()
        for (i, (name, center)) in model.parameters.enumerated() {
          var values = [Tensor]()
          for noise in subNoises {
            if let n = noise[i] {
              values.append(center.data! + n * eps)
              values.append(center.data! - n * eps)
            }
          }
          if !values.isEmpty { params[name] = Tensor(stack: values) }
        }
        let combinedMetrics = lossFn(subNoises.count * 2, params)
        #alwaysAssert(
          combinedMetrics.keys.contains(metric),
          "metric \(metric) not in returned metrics: \(combinedMetrics.keys)"
        )
        allMetrics.append(Metrics(dictionary: combinedMetrics.values.mapValues { $0.mean() }))

        for (i, noise) in subNoises.enumerated() {
          let posMetric = combinedMetrics[metric]![i * 2]
          let negMetric = combinedMetrics[metric]![i * 2 + 1]
          let delta = posMetric - negMetric
          let scale = delta / (2 * eps * Float(noises.count))
          try await scale.wait()

          for ((_, var param), noise) in zip(model.parameters, noise) {
            if let noise = noise {
              let g = noise * scale
              if let grad = param.grad { param.grad = grad + g } else { param.grad = g }
            }
          }
          allMetrics.append(Metrics(dictionary: [.named("es_delta"): delta]))
        }
      }

      return Metrics.mean(allMetrics)
    }
  }
}
