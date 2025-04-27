import ArgumentParser
import HCBacktrace
import Honeycrisp

public struct Graph: Sendable {
  public enum RandomKind: String, ExpressibleByArgument, CaseIterable, Sendable {
    case permutation
    case spatial
  }

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
    outCount: Int,
    kind: RandomKind = .permutation
  ) async throws -> Graph {
    switch kind {
    case .permutation:
      randomPermutation(
        batchSize: batchSize,
        cellCount: cellCount,
        actPerCell: actPerCell,
        inCount: inCount,
        outCount: outCount
      )
    case .spatial:
      try await randomSpatial(
        batchSize: batchSize,
        cellCount: cellCount,
        actPerCell: actPerCell,
        inCount: inCount,
        outCount: outCount
      )
    }
  }

  @recordCaller private static func _randomPermutation(
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

    let allPerms =
      Tensor(rand: [batchSize, cellCount, actPerCell]).argsort(axis: 1) * actPerCell
      + Tensor(data: 0..<actPerCell)

    return Graph(
      cellCount: cellCount,
      actPerCell: actPerCell,
      graphPerm: allPerms.flatten(startAxis: 1),
      inCount: inCount,
      outCount: outCount
    )
  }

  @recordCaller private static func _randomSpatial(
    batchSize: Int,
    cellCount: Int,
    actPerCell: Int,
    inCount: Int,
    outCount: Int,
    spaceDims: Int = 3
  ) async throws -> Graph {
    #alwaysAssert(
      cellCount >= inCount + outCount,
      "number of cells \(cellCount) must exceed inCount + outCount, which is \(inCount)+\(outCount) = \(inCount + outCount)"
    )

    var allGraphPerm = [Int]()

    for _ in 0..<batchSize {
      let coords = Tensor(randn: [cellCount, spaceDims])
      let pairwiseDists = (coords.unsqueeze(axis: 1) - coords).pow(2).sum(axis: -1)
      let sortedIndices = try await pairwiseDists.argsort(axis: 1).ints()

      var graphPerm = [Int](repeating: 0, count: cellCount * actPerCell)
      for actIdx in 0..<actPerCell {
        var hasInput = [Bool](repeating: false, count: cellCount)
        for cellIdx in Array(0..<cellCount).shuffled() {
          let neighbors = sortedIndices[(cellCount * cellIdx)..<(cellCount * (cellIdx + 1))]
          for neighbor in neighbors.makeIterator().dropFirst() {
            if !hasInput[neighbor] {
              graphPerm[cellIdx * actPerCell + actIdx] = neighbor * actPerCell + actIdx
              hasInput[neighbor] = true
              break
            }
          }
        }
      }
      allGraphPerm.append(contentsOf: graphPerm)
    }

    return Graph(
      cellCount: cellCount,
      actPerCell: actPerCell,
      graphPerm: Tensor(data: allGraphPerm, dtype: .int64).reshape([batchSize, -1]),
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
