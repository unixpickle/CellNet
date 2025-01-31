import ArgumentParser
import CellNet
import Foundation
import Honeycrisp
import MNIST

@main struct Main: AsyncParsableCommand {
  struct State: Codable {
    let model: Trainable.State
    let step: Int?
    let data: DataIterator.State?
    let opt: Adam.State?
    let clipper: GradClipper.State?
  }

  // Dataset configuration
  @Option(name: .long, help: "Dataset examples per rollout.") var examplesPerRollout: Int = 100
  @Option(name: .long, help: "Number of inference timesteps.") var inferSteps: Int = 5
  @Option(name: .long, help: "Number of update timesteps.") var updateSteps: Int = 5
  @Option(name: .shortAndLong, help: "Batch size.") var batchSize: Int = 2

  // Model hyperparameters
  @Option(name: .long, help: "State count.") var stateCount: Int = 64
  @Option(name: .long, help: "MLP hidden size.") var hiddenSize: Int = 64
  @Option(name: .long, help: "Number of cells.") var cellCount: Int = 1024
  @Option(name: .long, help: "Activations per cell.") var actPerCell: Int = 16

  // Other hyperparams
  @Option(name: .shortAndLong, help: "Learning rate.") var lr: Float = 0.001
  @Option(name: .long, help: "Adam beta1.") var beta1: Float = 0.9
  @Option(name: .long, help: "Adam beta2.") var beta2: Float = 0.999
  @Option(name: .long, help: "Adam epsilon.") var eps: Float = 1e-8
  @Option(name: .long, help: "AdamW weight decay.") var weightDecay: Float = 0.0

  // Saving
  @Option(name: .shortAndLong, help: "Output path.") var outputPath: String = "state.plist"
  @Option(name: .long, help: "Save interval.") var saveInterval: Int = 100

  mutating func run() async {
    print("Command:", CommandLine.arguments.joined(separator: " "))

    do {
      Backend.defaultBackend = try MPSBackend(allocator: .bucket)

      let cell = SyncTrainable(
        Cell(edgeCount: actPerCell, stateCount: stateCount, hiddenSize: hiddenSize)
      )
      let opt = Adam(
        cell.parameters,
        lr: lr,
        beta1: beta1,
        beta2: beta2,
        eps: eps,
        weightDecay: weightDecay
      )
      let clipper = GradClipper()

      let dataset = try await MNISTDataset.download(toDir: "mnist_data")
      var dataIt = DataIterator(images: dataset.train, batchSize: batchSize)

      var step: Int = 0

      if FileManager.default.fileExists(atPath: outputPath) {
        print("loading from checkpoint: \(outputPath) ...")
        let data = try Data(contentsOf: URL(fileURLWithPath: outputPath))
        let decoder = PropertyListDecoder()
        let state = try decoder.decode(State.self, from: data)
        try cell.loadState(state.model)
        if let optState = state.opt { try opt.loadState(optState) }
        if let clipperState = state.clipper { clipper.state = clipperState }
        if let dataState = state.data { dataIt.state = dataState }
        step = state.step ?? 0
      }

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
            ) * 2 - 1
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
          logits.logSoftmax(axis: -1).gather(axis: 1, indices: labelIdxs[..., NewAxis()]).mean()
        }
        let accs = zip(allLabelIndices, rollouts.outputs).map { (labelIdxs, logits) in
          (labelIdxs == logits.argmax(axis: 1)).cast(.float32).mean()
        }
        let meanLoss = -Tensor(stack: losses).mean()
        let meanAcc = Tensor(stack: accs).mean()

        meanLoss.backward()

        let (gradNorm, gradScale) = try await clipper.clipGrads(model: cell)

        opt.step()
        opt.clearGrads()

        step += 1
        print(
          "step \(step):" + " loss=\(try await meanLoss.item())"
            + " acc=\(try await meanAcc.item())" + " grad_norm=\(gradNorm) grad_scale=\(gradScale)"
        )

        if step % saveInterval == 0 {
          print("saving after \(step) steps...")
          let state = State(
            model: try await cell.state(),
            step: step,
            data: dataIt.state,
            opt: try await opt.state(),
            clipper: clipper.state
          )
          let stateData = try PropertyListEncoder().encode(state)
          try stateData.write(to: URL(filePath: outputPath), options: .atomic)
        }
      }
    } catch { print("FATAL ERROR: \(error)") }
  }
}
