import ArgumentParser
import CellNet
import Foundation
import HCBacktrace
import Honeycrisp
import MNIST

@main struct Main: AsyncParsableCommand {
  typealias LossAndAcc = (loss: Tensor, acc: Tensor)

  struct State: Codable {
    let model: Trainable.State
    let step: Int?
    let data: DataIterator.State?
    let opt: Muon.State?
    let clipper: GradClipper.State?
  }

  // Dataset configuration
  @Option(name: .long, help: "Dataset examples per rollout.") var examplesPerRollout: Int = 100
  @Option(name: .long, help: "Number of inference timesteps.") var inferSteps: Int = 5
  @Option(name: .long, help: "Number of update timesteps.") var updateSteps: Int = 5
  @Option(name: .shortAndLong, help: "Batch size.") var batchSize: Int = 2
  @Option(name: .long, help: "Divide batches into microbatches.") var microbatch: Int? = nil

  // Model hyperparameters
  @Option(name: .long, help: "State count.") var stateCount: Int = 64
  @Option(name: .long, help: "MLP hidden size.") var hiddenSize: Int = 64
  @Option(name: .long, help: "Number of cells.") var cellCount: Int = 1024
  @Option(name: .long, help: "Activations per cell.") var actPerCell: Int = 16
  @Option(name: .long, help: "Normalization mode.") var normalization: Cell.Normalization = .none

  // Muon hyperparams
  @Option(name: .shortAndLong, help: "Learning rate.") var lr: Float = 0.01
  @Option(name: .long, help: "Muon momentum.") var momentum: Float = 0.95
  @Option(name: .long, help: "Muon weight decay.") var weightDecay: Float = 0.0

  // Other optimizer hyperparameters
  @Option(name: .long, help: "Maximum gradient RMS before clipping.") var maxGradRMS: Float = 100.0

  // Evolutions strategies
  @Option(name: .long, help: "If specified, use Evolution Strategies with this epsilon.")
  var esEpsilon: Float? = nil
  @Option(name: .long, help: "The ES population size.") var esPopulation: Int = 1

  // Saving
  @Option(name: .shortAndLong, help: "Output path.") var outputPath: String = "state.plist"
  @Option(name: .long, help: "Save interval.") var saveInterval: Int = 100

  mutating func run() async {
    print("Command:", CommandLine.arguments.joined(separator: " "))

    do {
      Backend.defaultBackend = try MPSBackend(allocator: .bucket)

      let cell = SyncTrainable(
        Cell(
          edgeCount: actPerCell,
          stateCount: stateCount,
          hiddenSize: hiddenSize,
          normalization: normalization
        )
      )
      let opt = Muon(cell.parameters, lr: lr, momentum: momentum, weightDecay: weightDecay)
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

      let es: ES? =
        if let eps = esEpsilon { ES(model: cell, eps: eps, directionCount: esPopulation) } else {
          nil
        }

      while true {
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

        var allMetrics = [Metrics]()
        let mbSize = microbatch ?? batchSize
        let esNoise = es?.sampleNoises()
        let actGradClipper = ActGradClipper(maxRMS: maxGradRMS)

        for i in stride(from: 0, to: batchSize, by: mbSize) {
          let curMbSize = min(batchSize - i, mbSize)
          let mbScale = Float(curMbSize) / Float(batchSize)
          let graph = Graph.random(
            batchSize: curMbSize,
            cellCount: cellCount,
            actPerCell: actPerCell,
            inCount: 28 * 28,
            outCount: 10
          )

          func computeLosses() -> Metrics {
            let rollouts = Rollout.rollout(
              inferSteps: inferSteps,
              updateSteps: updateSteps,
              inputs: allInputs.map { $0[i..<(i + curMbSize)] },
              targets: allLabels.map { $0[i..<(i + curMbSize)] },
              cell: cell,
              graph: graph,
              clipper: actGradClipper
            )

            let subLabelIndices = allLabelIndices.map { $0[i..<(i + curMbSize)] }
            let losses = zip(subLabelIndices, rollouts.outputs).map { (labelIdxs, logits) in
              logits.logSoftmax(axis: -1).gather(axis: 1, indices: labelIdxs[..., NewAxis()]).mean()
            }
            let accs = zip(subLabelIndices, rollouts.outputs).map { (labelIdxs, logits) in
              (labelIdxs == logits.argmax(axis: 1)).cast(.float32).mean()
            }
            let meanLoss = -Tensor(stack: losses).mean() * mbScale
            let meanAcc = Tensor(stack: accs).mean() * mbScale
            return [.loss: meanLoss, .accuracy: meanAcc]
          }

          let metrics =
            if let es = es, let esNoise = esNoise {
              es.forwardBackward(noises: esNoise, lossFn: computeLosses)
            } else {
              try await {
                let metrics = computeLosses()
                metrics[.loss]!.backward()
                // Wait for backward computation before using memory
                // for the next microbatch.
                if i + curMbSize < batchSize {
                  for (_, p) in cell.parameters { if let g = p.grad { try await g.wait() } }
                }
                return metrics.noGrad()
              }()
            }
          allMetrics.append(metrics)
        }

        let (gradNorm, gradScale) = try await clipper.clipGrads(model: cell)

        opt.step()
        opt.clearGrads()

        step += 1
        var metrics = Metrics.sum(allMetrics)
        metrics += actGradClipper.metrics
        metrics[.gradNorm] = Tensor(data: [gradNorm])
        metrics[.gradScale] = Tensor(data: [gradScale])
        print("step \(step): \(try await metrics.format())")

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
