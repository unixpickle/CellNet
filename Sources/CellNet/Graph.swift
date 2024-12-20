import HCBacktrace
import Honeycrisp

public struct Graph: Sendable {
  public let batchSize: Int
  public let cellCount: Int
  public let actPerCell: Int

  /// A permutation of shape [batchSize, actCount].
  public let graphPerm: Tensor

  /// Indices of shape [batchSize, inCount] with integer elements in [0, actCount).
  public let inputIndices: Tensor

  /// Indices of shape [batchSize, outCount] with integer elements in [0, actCount).
  public let targetIndices: Tensor

  public var actCount: Int { cellCount * actPerCell }
  public var inCount: Int { inputIndices.shape[0] }
  public var outCount: Int { targetIndices.shape[0] }

  public init(
    batchSize: Int,
    cellCount: Int,
    actPerCell: Int,
    graphPerm: Tensor,
    inputIndices: Tensor,
    targetIndices: Tensor
  ) {
    self.batchSize = batchSize
    self.cellCount = cellCount
    self.actPerCell = actPerCell
    self.graphPerm = graphPerm
    self.inputIndices = inputIndices
    self.targetIndices = targetIndices
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
    var allInputIndices = [Tensor]()
    var allTargetIndices = [Tensor]()

    for _ in 0..<batchSize {
      var graphPerm = [Int](repeating: 0, count: cellCount * actPerCell)
      for i in 0..<actPerCell {
        var permForTerminal = Array(0..<cellCount)
        permForTerminal.shuffle()
        for (j, k) in permForTerminal.enumerated() {
          graphPerm[j * actPerCell + i] = k * actPerCell + i
        }
      }
      let inputIndices = Array(stride(from: 0, to: inCount * actPerCell, by: actPerCell))
      let targetIndices = Array(
        stride(
          from: inCount * actPerCell + 1,
          to: (inCount + outCount) * actPerCell + 1,
          by: actPerCell
        )
      )
      allGraphPerm.append(Tensor(data: graphPerm, dtype: .int64))
      allInputIndices.append(Tensor(data: inputIndices, dtype: .int64))
      allTargetIndices.append(Tensor(data: targetIndices, dtype: .int64))
    }

    return Graph(
      batchSize: batchSize,
      cellCount: cellCount,
      actPerCell: actPerCell,
      graphPerm: Tensor(stack: allGraphPerm),
      inputIndices: Tensor(stack: allInputIndices),
      targetIndices: Tensor(stack: allTargetIndices)
    )
  }

  /// Gather the outputs from the activation tensor.
  ///
  /// This should be done before outputsToInputs().
  public func gatherOutputs(activations: Tensor) -> Tensor {
    activations.gather(axis: 1, indices: targetIndices)
  }

  /// Take outputs from the last step and turn them into inputs
  /// for the next step.
  public func outputsToInputs(activations: Tensor) -> Tensor {
    return activations.gather(axis: 1, indices: graphPerm)
  }

  /// Replace some of the activations with inputs.
  ///
  /// This should be done after outputsToInputs().
  public func populateInputs(activations: Tensor, inputs: Tensor) -> Tensor {
    let mask = Tensor(onesLike: inputs).cast(.bool).scatter(
      axis: 1,
      count: actCount,
      indices: inputIndices
    )
    let scattered = inputs.scatter(axis: 1, count: actCount, indices: inputIndices)
    return mask.when(isTrue: scattered, isFalse: activations)
  }

  /// Replace some of the activations with targets.
  ///
  /// This should be done after outputsToInputs().
  public func populateTargets(activations: Tensor, targets: Tensor) -> Tensor {
    let mask = Tensor(onesLike: targets).cast(.bool).scatter(
      axis: 1,
      count: actCount,
      indices: targetIndices
    )
    let scattered = targets.scatter(axis: 1, count: actCount, indices: targetIndices)
    return mask.when(isTrue: scattered, isFalse: activations)
  }
}
