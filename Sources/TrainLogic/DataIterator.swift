import Honeycrisp

struct DataIterator: Sequence, IteratorProtocol {
  let batchSize: Int
  let exampleCount: Int
  var offset = 0

  mutating func next() -> ([Tensor], [Tensor])? {
    var allInputs = [Tensor]()
    var allLabels = [Tensor]()
    let gates = (0..<batchSize).map { _ in Int.random(in: 0..<3) }
    for _ in 0..<exampleCount {
      var inputData = [Float]()
      var outputLabels = [Int]()
      for gate in gates {
        let in1 = Int.random(in: 0...1)
        let in2 = Int.random(in: 0...1)
        let out =
          switch gate {
          case 0: in1 | in2
          case 1: in1 & in2
          case 2: in1 ^ in2
          default: fatalError()
          }
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
