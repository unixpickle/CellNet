import HCBacktrace
import Honeycrisp

public struct Rollout {
  public let inferSteps: Int
  public let updateSteps: Int

  public let inputs: [Tensor]
  public let targets: [Tensor]
  public let outputs: [Tensor]

  public init(
    inferSteps: Int,
    updateSteps: Int,
    inputs: [Tensor],
    targets: [Tensor],
    outputs: [Tensor]
  ) {
    self.inferSteps = inferSteps
    self.updateSteps = updateSteps
    self.inputs = inputs
    self.targets = targets
    self.outputs = outputs
  }

  @recordCaller private static func _rollout(
    inferSteps: Int,
    updateSteps: Int,
    inputs: [Tensor],
    targets: [Tensor],
    cell: SyncTrainable<Cell>,
    graph: Graph,
    resetActs: Bool = false
  ) -> Rollout {
    #alwaysAssert(inputs.count == targets.count)
    let batchShape = inputs[0].shape[..<(inputs[0].shape.count - 1)]
    var state = NetworkState(
      inputs: Tensor(zeros: batchShape + [graph.cellCount]),
      targets: Tensor(zeros: batchShape + [graph.cellCount]),
      prevActivations: Tensor(zeros: batchShape + [graph.actCount]),
      activations: Tensor(zeros: batchShape + [graph.actCount]),
      cellStates: Tensor(
        zeros: batchShape + [graph.cellCount * cell.use { cell in cell.stateCount }]
      )
    )

    func checkpointedCell(_ state: NetworkState) -> (outputs: Tensor, state: NetworkState) {
      let results = Tensor.checkpoint([
        state.inputs, state.targets, state.prevActivations, state.activations, state.cellStates,
      ]) { (xs: [Tensor]) -> [Tensor] in
        let results = cell.use { cell in
          cell(
            NetworkState(
              inputs: xs[0],
              targets: xs[1],
              prevActivations: xs[2],
              activations: xs[3],
              cellStates: xs[4]
            )
          )
        }
        return [results.outputs, results.newActs, results.newCellStates]
      }
      return (
        outputs: results[0],
        state: state.with(
          prevActivations: results[1],
          activations: results[1],
          cellStates: results[2]
        )
      )
    }

    var outputs = [Tensor]()
    for (i, (input, target)) in zip(inputs, targets).enumerated() {
      if i > 0 && resetActs {
        state = state.with(
          prevActivations: Tensor(zerosLike: state.prevActivations),
          activations: Tensor(zerosLike: state.activations)
        )
      }
      state = state.with(inputs: graph.populateInputs(inputs: input))
      for step in 0..<inferSteps {
        let out = checkpointedCell(state)
        state = out.state
        if step + 1 == inferSteps { outputs.append(graph.gatherOutputs(outputs: out.outputs)) }
        state = state.with(activations: graph.stepOutToStepIn(activations: state.activations))
      }
      state = state.with(targets: graph.populateTargets(targets: target))

      if i + 1 == inputs.count { break }

      for _ in 0..<updateSteps {
        state = checkpointedCell(state).state
        state = state.with(activations: graph.stepOutToStepIn(activations: state.activations))
      }
    }

    return Rollout(
      inferSteps: inferSteps,
      updateSteps: updateSteps,
      inputs: inputs,
      targets: targets,
      outputs: outputs
    )
  }
}
