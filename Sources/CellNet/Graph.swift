import HCBacktrace
import Honeycrisp

public struct Graph: Sendable {
  public let cellCount: Int
  public let actPerCell: Int
  public let inCount: Int
  public let outCount: Int

  /// A permutation of shape [batchSize, actCount].
  public let graphPerm: Tensor

  public var batchSize: Int { graphPerm.shape[0] }
  public var actCount: Int { cellCount * actPerCell }

  public init(cellCount: Int, actPerCell: Int, graphPerm: Tensor, inCount: Int, outCount: Int) {
    self.cellCount = cellCount
    self.actPerCell = actPerCell
    self.graphPerm = graphPerm
    self.inCount = inCount
    self.outCount = outCount
  }

  @recordCaller private static func _random(
    batchSize: Int,
    cellCount: Int,
    actPerCell: Int,
    inCount: Int,
    outCount: Int
  ) -> Graph {
    #alwaysAssert(
      cellCount >= inCount + outCount,
      "number of cells \(cellCount) must exceed inCount + outCount, which is \(inCount)+\(outCount) = \(inCount + outCount)"
    )
    #alwaysAssert(
      actPerCell >= 2,
      "must have at least two activations per cell: one for inputs, one for targets"
    )

    var allGraphPerm = [Tensor]()

    for _ in 0..<batchSize {
      var graphPerm = [Int](repeating: 0, count: cellCount * actPerCell)
      for i in 0..<actPerCell {
        var permForTerminal = Array(0..<cellCount)
        permForTerminal.shuffle()
        for (j, k) in permForTerminal.enumerated() {
          graphPerm[j * actPerCell + i] = k * actPerCell + i
        }
      }
      allGraphPerm.append(Tensor(data: graphPerm, dtype: .int64))
    }

    return Graph(
      cellCount: cellCount,
      actPerCell: actPerCell,
      graphPerm: Tensor(stack: allGraphPerm),
      inCount: inCount,
      outCount: outCount
    )
  }

  @recordCaller private func _repeated(count: Int) -> Graph {
    if count == 1 {
      self
    } else {
      Graph(
        cellCount: cellCount,
        actPerCell: actPerCell,
        graphPerm: graphPerm.repeating(axis: 0, count: count),
        inCount: inCount,
        outCount: outCount
      )
    }
  }

  /// Gather the used outputs from the outputs tensor.
  public func gatherOutputs(outputs: Tensor) -> Tensor {
    return outputs[..., inCount..<(inCount + outCount)]
  }

  /// Take output activations from the last step and turn them into inputs
  /// for the next step.
  public func stepOutToStepIn(activations: Tensor) -> Tensor {
    return activations.gather(axis: 1, indices: graphPerm)
  }

  /// Create an input tensor.
  public func populateInputs(inputs: Tensor) -> Tensor {
    return Tensor(
      concat: [inputs, Tensor(zeros: [inputs.shape[0], cellCount - inCount])],
      axis: -1
    )
  }

  /// Create a target tensor.
  public func populateTargets(targets: Tensor) -> Tensor {
    return Tensor(
      concat: [
        Tensor(zeros: [targets.shape[0], inCount]), targets,
        Tensor(zeros: [targets.shape[0], cellCount - (inCount + outCount)]),
      ],
      axis: -1
    )
  }
}
