import CellNet
import Honeycrisp
import MNIST

@main struct Main {
  // Dataset configuration
  static let examplesPerRollout: Int = 20
  static let inferSteps: Int = 5
  static let updateSteps: Int = 5
  static let batchSize: Int = 2

  // Model hyperparameters
  static let stateCount: Int = 8
  static let hiddenSize: Int = 32
  static let cellCount: Int = 1024
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

      let dataset = try await MNISTDataset.download(toDir: "mnist_data")
      var dataIt = DataIterator(images: dataset.train.shuffled(), batchSize: batchSize)

      var step: Int = 0
      while true {
        let graph = Graph.random(
          batchSize: batchSize,
          cellCount: cellCount,
          actPerCell: actPerCell,
          inCount: 28 * 28,
          outCount: 10
        )

        var allInputs = [Tensor]()
        var allLabelIndices = [Tensor]()
        var allLabels = [Tensor]()
        for _ in 0..<(examplesPerRollout) {
          let (inputs, labels) = dataIt.next()!
          allInputs.append(inputs)
          allLabelIndices.append(labels)
          allLabels.append(
            Tensor(ones: [labels.shape[0], 1]).scatter(
              axis: 1,
              count: 10,
              indices: labels[..., NewAxis()]
            )
          )
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
          logits.logSoftmax(axis: -1).gather(axis: 1, indices: labelIdxs).mean()
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
