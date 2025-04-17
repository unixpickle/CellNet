import ArgumentParser
import CellNet
import DataUtils
import Foundation
import HCBacktrace
import Honeycrisp

@main struct Main: AsyncParsableCommand {

  struct State: Codable {
    let model: Trainable.State
    let step: Int?
    let data: MNISTIterator.State?
    let opt: Muon.State?
    let clipper: GradClipper.State?
  }

  // Dataset configuration
  @Option(name: .long, help: "Dataset source.") var datasetSource: MNISTIterator.Source = .mnist
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
  @Option(name: .long, help: "Activation scale.") var actScale: Float = 1.0
  @Option(name: .long, help: "Initialization scheme.") var initialization:
    MetaLinear.Initialization = .xavier

  // Muon hyperparams
  @Option(name: .shortAndLong, help: "Learning rate.") var lr: Float = 0.01
  @Option(name: .shortAndLong, help: "Minimum eigenvalue to rescale in Muon.") var muonEpsilon:
    Float = 0.01
  @Option(name: .long, help: "Muon momentum.") var momentum: Float = 0.95
  @Option(name: .long, help: "Muon weight decay.") var weightDecay: Float = 0.0

  // Evolutions strategies
  @Option(name: .long, help: "If specified, use Evolution Strategies with this epsilon.")
  var esEpsilon: Float? = nil
  @Option(name: .long, help: "The ES population size.") var esPopulation: Int = 1
  @Option(name: .long, help: "The ES batch size.") var esBatchSize: Int = 1

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
          normalization: normalization,
          initialization: initialization,
          actScale: actScale
        )
      )
      func waitForGrads() async throws {
        for (_, p) in cell.parameters { if let g = p.grad { try await g.wait() } }
      }

      let opt = Muon(
        cell.parameters,
        lr: lr,
        momentum: momentum,
        weightDecay: weightDecay,
        nsSteps: nil,
        epsilon: muonEpsilon
      )
      let clipper = GradClipper()

      var dataIt = try await MNISTIterator(batchSize: batchSize, source: datasetSource)

      var step: Int = 0

      let es: ES? =
        if let eps = esEpsilon { ES(model: cell, eps: eps, directionCount: esPopulation) } else {
          nil
        }

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
        let (allInputs, allLabelIndices, allLabels) = dataIt.rollout(count: examplesPerRollout)!

        let esNoise = es?.sampleNoises()

        func computeGradients() async throws -> Metrics {
          var allMetrics = [Metrics]()
          let mbSize = microbatch ?? batchSize
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

            func computeLosses(
              count: Int = 1,
              params: [String: Tensor]? = nil,
              scalarMetrics: Bool = false
            ) -> Metrics {
              let rollouts = Rollout.rollout(
                inferSteps: inferSteps,
                updateSteps: updateSteps,
                inputs: allInputs.map { $0[i..<(i + curMbSize)].repeating(axis: 0, count: count) },
                targets: allLabels.map { $0[i..<(i + curMbSize)].repeating(axis: 0, count: count) },
                cell: cell,
                graph: graph.repeated(count: count),
                params: params
              )

              // Tiny computations are faster on CPU and don't launch a ton of kernels.
              return CPUBackend.global.use {
                func means(_ x: Tensor) -> Tensor {
                  if scalarMetrics { return x.mean() }
                  let x = x.reshape([count, x.shape[0] / count] + x.shape[1...])
                  return x.flatten(startAxis: 1).mean(axis: 1)
                }

                let subLabelIndices = allLabelIndices.map {
                  $0[i..<(i + curMbSize)].repeating(axis: 0, count: count)
                }
                let losses = zip(subLabelIndices, rollouts.outputs).map { (labelIdxs, logits) in
                  means(
                    logits.logSoftmax(axis: -1).gather(axis: 1, indices: labelIdxs[..., NewAxis()])
                  )
                }
                let accs = zip(subLabelIndices, rollouts.outputs).map { (labelIdxs, logits) in
                  means((labelIdxs == logits.argmax(axis: 1)).cast(.float32))
                }

                let meanLoss = -Tensor(stack: losses).mean(axis: 0) * mbScale
                let meanAcc = Tensor(stack: accs).mean(axis: 0) * mbScale
                return [
                  .loss: meanLoss, .accuracy: meanAcc, .lastLoss: -losses.last!.noGrad() * mbScale,
                  .lastAccuracy: accs.last! * mbScale,
                ]
              }
            }

            let metrics =
              if let es = es, let esNoise = esNoise {
                try await es.forwardBackward(noises: esNoise, batchSize: esBatchSize) { x, y in
                  computeLosses(count: x, params: y)
                }
              } else {
                try await {
                  let metrics = computeLosses(scalarMetrics: true)
                  metrics[.loss]!.backward()
                  try await waitForGrads()
                  return metrics.noGrad()
                }()
              }
            allMetrics.append(metrics)
          }
          return Metrics.sum(allMetrics)
        }

        var metrics = try await computeGradients()

        let (gradNorm, gradScale) = try await clipper.clipGrads(model: cell)

        opt.step()
        opt.clearGrads()

        step += 1
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
