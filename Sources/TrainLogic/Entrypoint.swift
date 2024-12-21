import CellNet
import Honeycrisp

@main struct Main {
  // Dataset configuration
  static let examplesPerRollout: Int = 5
  static let inferSteps: Int = 3
  static let updateSteps: Int = 3
  static let batchSize: Int = 8

  // Model hyperparameters
  static let stateCount: Int = 8
  static let hiddenSize: Int = 64
  static let cellCount: Int = 16
  static let actPerCell: Int = 8

  // Other hyperparams
  static let lr: Float = 0.001

  static func main() async {
    do {
      Backend.defaultBackend = try MPSBackend(allocator: .bucket)

      let cell = SyncTrainable(
        Cell(edgeCount: actPerCell, stateCount: stateCount, hiddenSize: hiddenSize)
      )
      let opt = Adam(cell.parameters, lr: lr)
      let dataIt = DataIterator(batchSize: batchSize, exampleCount: examplesPerRollout)

      var step: Int = 0
      for (allInputs, allLabelIndices) in dataIt {
        let graph = Graph.random(
          batchSize: batchSize,
          cellCount: cellCount,
          actPerCell: actPerCell,
          inCount: 4,
          outCount: 2
        )
        let allLabels = allLabelIndices.map { idxs in
          Tensor(ones: [idxs.shape[0], 1]).scatter(axis: 1, count: 2, indices: idxs[..., NewAxis()])
        }

        let rollouts = Rollout.rollout(
          inferSteps: inferSteps,
          updateSteps: updateSteps,
          inputs: allInputs,
          targets: allLabels,
          cell: cell,
          graph: graph
        )

        let losses = zip(allLabelIndices, rollouts.outputs).map { (labelIdxs, logits) in
          logits.logSoftmax(axis: -1).gather(axis: 1, indices: labelIdxs[..., NewAxis()]).mean()
        }
        let accs = zip(allLabelIndices, rollouts.outputs).map { (labelIdxs, logits) in
          (labelIdxs == logits.argmax(axis: 1)).cast(.float32).mean()
        }
        let meanLoss = -Tensor(stack: losses).mean()
        let meanAcc = Tensor(stack: accs).mean()

        meanLoss.backward()

        var gradNorm = Tensor(zeros: [])
        for (_, p) in cell.parameters { gradNorm = gradNorm + p.grad!.pow(2).sum() }

        opt.step()
        opt.clearGrads()

        step += 1
        print(
          "step \(step):" + " loss=\(try await meanLoss.item())"
            + " acc=\(try await meanAcc.item())" + " grad_norm=\(try await gradNorm.sqrt().item())"
        )
      }
    } catch { print("FATAL ERROR: \(error)") }
  }
}
