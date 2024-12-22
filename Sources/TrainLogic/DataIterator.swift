import Honeycrisp

enum LogicGate {
  case or
  case and
  case xor

  static func allGates() -> [LogicGate] { [Self.or, Self.and, Self.xor] }
  static func xorOnly() -> [LogicGate] { [Self.xor] }

  func apply(_ x: Int, _ y: Int) -> Int {
    switch self {
    case .or: x | y
    case .and: x & y
    case .xor: x ^ y
    }
  }
}

struct DataIterator: Sequence, IteratorProtocol {
  let batchSize: Int
  let exampleCount: Int
  let allowedGates: [LogicGate]
  var offset = 0

  mutating func next() -> ([Tensor], [Tensor])? {
    var allInputs = [Tensor]()
    var allLabels = [Tensor]()
    let gates = (0..<batchSize).map { _ in allowedGates.randomElement()! }
    for _ in 0..<exampleCount {
      var inputData = [Float]()
      var outputLabels = [Int]()
      for gate in gates {
        let in1 = Int.random(in: 0...1)
        let in2 = Int.random(in: 0...1)
        let out = gate.apply(in1, in2)
        inputData.append(Float(in1))
        inputData.append(1 - Float(in1))
        inputData.append(Float(in2))
        inputData.append(1 - Float(in2))
        outputLabels.append(out)
      }
      allInputs.append(Tensor(data: inputData, shape: [batchSize, 4]))
      allLabels.append(Tensor(data: outputLabels, shape: [batchSize]))
    }
    return (allInputs, allLabels)
  }
}
