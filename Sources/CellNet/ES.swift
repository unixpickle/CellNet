import HCBacktrace
import Honeycrisp

/// Evolution strategies implementation.
public class ES {
  public typealias LossAndMetrics = (loss: Tensor, metrics: [String: Tensor])

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

  @recordCaller private func _forwardBackward(noises: [[Tensor?]], lossFn: () -> LossAndMetrics)
    -> LossAndMetrics
  {
    Tensor.withGrad(enabled: false) {
      var allLosses = [Tensor]()
      var allMetrics = [String: [Tensor]]()
      let center = model.parameters.map { $0.1.data }

      for noise in noises {
        func assignWithScale(_ scale: Float) {
          for ((_, var param), (center, noise)) in zip(model.parameters, zip(center, noise)) {
            if let center = center, let noise = noise { param.data = center + noise * scale }
          }
        }
        assignWithScale(-eps)
        let (negLoss, negMetrics) = lossFn()
        assignWithScale(eps)
        let (posLoss, posMetrics) = lossFn()
        assignWithScale(0.0)

        for ((_, var param), noise) in zip(model.parameters, noise) {
          if let noise = noise {
            let g = noise * (posLoss - negLoss) / (2 * eps * Float(noises.count))
            if let grad = param.grad { param.grad = grad + g } else { param.grad = g }
          }
        }
        allLosses.append(contentsOf: [negLoss, posLoss])
        for d in [negMetrics, posMetrics] {
          for (k, v) in d { allMetrics[k] = (allMetrics[k] ?? []) + [v] }
        }
      }

      return (loss: averageMetric(allLosses), metrics: allMetrics.mapValues { averageMetric($0) })
    }
  }
}

func averageMetric(_ values: [Tensor]) -> Tensor { return Tensor(stack: values).mean() }
