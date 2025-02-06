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
    lossFn: () -> Metrics
  ) -> Metrics {
    Tensor.withGrad(enabled: false) {
      var allMetrics = [Metrics]()
      let center = model.parameters.map { $0.1.data }

      for noise in noises {
        func assignWithScale(_ scale: Float) {
          for ((_, var param), (center, noise)) in zip(model.parameters, zip(center, noise)) {
            if let center = center, let noise = noise { param.data = center + noise * scale }
          }
        }
        assignWithScale(-eps)
        let negMetrics = lossFn()
        assignWithScale(eps)
        let posMetrics = lossFn()
        assignWithScale(0.0)

        #alwaysAssert(
          negMetrics.keys.contains(metric),
          "metric \(metric) not in returned metrics: \(negMetrics.keys)"
        )
        #alwaysAssert(
          posMetrics.keys.contains(metric),
          "metric \(metric) not in returned metrics: \(posMetrics.keys)"
        )

        let delta = posMetrics[metric]! - negMetrics[metric]!
        let scale = delta / (2 * eps * Float(noises.count))

        for ((_, var param), noise) in zip(model.parameters, noise) {
          if let noise = noise {
            let g = noise * scale
            if let grad = param.grad { param.grad = grad + g } else { param.grad = g }
          }
        }
        allMetrics.append(contentsOf: [negMetrics, posMetrics])
        allMetrics.append(Metrics(dictionary: [.esDelta: delta]))
      }

      return Metrics.mean(allMetrics)
    }
  }
}
