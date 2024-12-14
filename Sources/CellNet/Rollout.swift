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
    graph: Graph
  ) -> Rollout {
    #alwaysAssert(inputs.count == targets.count)
    let batchShape = inputs[0].shape[..<(inputs[0].shape.count - 1)]
    var state = NetworkState(
      activations: Tensor(zeros: batchShape + [graph.actCount]),
      cellStates: Tensor(
        zeros: batchShape + [graph.cellCount * cell.use { cell in cell.stateCount }]
      )
    )

    func checkpointedCell(_ state: NetworkState) -> NetworkState {
      let results = Tensor.checkpoint([state.activations, state.cellStates]) {
        (xs: [Tensor]) -> [Tensor] in
        let results = cell.use { cell in cell(NetworkState(activations: xs[0], cellStates: xs[1])) }
        return [results.activations, results.cellStates]
      }
      return NetworkState(activations: results[0], cellStates: results[1])
    }

    var outputs = [Tensor]()
    for (i, (input, target)) in zip(inputs, targets).enumerated() {
      for step in 0..<inferSteps {
        state = state.with(
          activations: graph.populateInputs(activations: state.activations, inputs: input)
        )
        state = checkpointedCell(state)
        if step + 1 == inferSteps {
          outputs.append(graph.gatherOutputs(activations: state.activations))
        }
        state = state.with(activations: graph.outputsToInputs(activations: state.activations))
      }

      if i + 1 == inputs.count { break }

      for _ in 0..<updateSteps {
        state = state.with(
          activations: graph.populateTargets(
            activations: graph.populateInputs(activations: state.activations, inputs: input),
            targets: target
          )
        )
        state = checkpointedCell(state)
        state = state.with(activations: graph.outputsToInputs(activations: state.activations))
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
