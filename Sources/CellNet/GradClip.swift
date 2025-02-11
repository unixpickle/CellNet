import Foundation
import HCBacktrace
import Honeycrisp

public final class ActGradClipper: Sendable {
  public let maxRMS: Float

  let lock = NSLock()

  nonisolated(unsafe) var _rms: [Tensor] = []
  nonisolated(unsafe) var _clipped: [Tensor] = []

  public var rms: [Tensor] { lock.withLock { _rms } }
  public var clipped: [Tensor] { lock.withLock { _clipped } }
  public var metrics: Metrics {
    if rms.isEmpty {
      Metrics()
    } else {
      Metrics(dictionary: [
        .actGradRMS: Tensor(stack: rms).mean(),
        .actGradClipFrac: Tensor(stack: clipped).cast(.float32).mean(),
      ])
    }
  }

  public init(maxRMS: Float) { self.maxRMS = maxRMS }

  func add(rms: Tensor, clipped: Tensor) {
    lock.withLock {
      _rms.append(rms)
      _clipped.append(clipped)
    }
  }

  @recordCaller private func _callAsFunction(_ x: Tensor) -> Tensor {
    if !x.needsGrad || !Tensor.isGradEnabled { return x }
    let backend = Backend.current
    let handle = x.saveForBackward()
    return x.noGrad().onGrad { [self] g in
      let flatG = g.flatten(startAxis: 1)
      let rms = flatG.pow(2).mean(axis: 1).sqrt()
      let scales = (maxRMS / rms).clamp(max: 1.0)
      self.add(rms: rms, clipped: scales < 1)
      handle.backward(backend) { (flatG * scales[..., NewAxis()]).reshape(as: g) }
    }
  }
}
