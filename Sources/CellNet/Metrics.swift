import HCBacktrace
import Honeycrisp

public struct Metrics: ExpressibleByDictionaryLiteral, Sequence {
  public typealias Value = Tensor

  public enum Key {
    case loss
    case accuracy
    case gradNorm
    case gradScale
    case esDelta
    case actGradClipFrac
    case actGradRMS

    public var logKey: String {
      switch self {
      case .loss: "loss"
      case .accuracy: "acc"
      case .gradNorm: "grad_norm"
      case .gradScale: "grad_scale"
      case .esDelta: "es_delta"
      case .actGradClipFrac: "act_grad_clip_frac"
      case .actGradRMS: "act_grad_rms"
      }
    }
  }

  var values: [Key: Tensor]
  public var keys: [Key] { Array(values.keys) }

  public init() { values = [:] }

  public init(dictionary: [Key: Tensor]) { values = dictionary }

  public init(dictionaryLiteral elements: (Key, Value)...) {
    values = Dictionary(uniqueKeysWithValues: elements)
  }

  public init(uniqueKeysWithValues elements: [(Key, Value)]) {
    values = Dictionary(uniqueKeysWithValues: elements)
  }

  public func makeIterator() -> Dictionary<Key, Tensor>.Iterator { values.makeIterator() }

  public subscript(_ key: Key) -> Tensor? {
    get { values[key] }
    set { values[key] = newValue }
  }

  public func noGrad() -> Metrics { return Metrics(dictionary: values.mapValues { $0.noGrad() }) }

  public static func += (lhs: inout Metrics, rhs: Metrics) {
    for (k, v) in rhs {
      #alwaysAssert(lhs[k] == nil, "duplicate key \(k) when combining metrics")
      lhs[k] = v
    }
  }

  @recordCaller private static func _mean(_ metrics: [Metrics]) -> Metrics {
    let keys = Set(metrics.flatMap { $0.keys })
    return Metrics(
      uniqueKeysWithValues: keys.map { key in
        let values = metrics.map { $0[key] }.filter { $0 != nil }.map { $0! }
        return (key, Tensor(stack: values).mean())
      }
    )
  }

  @recordCaller private static func _sum(_ metrics: [Metrics]) -> Metrics {
    let keys = Set(metrics.flatMap { $0.keys })
    return Metrics(
      uniqueKeysWithValues: keys.map { key in
        let values = metrics.map { $0[key] }.filter { $0 != nil }.map { $0! }
        return (key, Tensor(stack: values).sum())
      }
    )
  }

  @recordCaller private func _format() async throws -> String {
    var fields = [String]()
    for (key, value) in self { fields.append("\(key.logKey)=\(try await value.item())") }
    fields.sort()
    return fields.joined(separator: " ")
  }
}
