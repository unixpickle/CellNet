import ArgumentParser
import CMA
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
    let cma: CMA.State?
  }

  // Dataset configuration
  @Option(name: .long, help: "Dataset examples per rollout.") var examplesPerRollout: Int = 100
  @Option(name: .long, help: "Number of inference timesteps.") var inferSteps: Int = 5
  @Option(name: .long, help: "Number of update timesteps.") var updateSteps: Int = 5
  @Option(name: .shortAndLong, help: "Batch size.") var batchSize: Int = 8
  @Option(name: .long, help: "Divide batches into microbatches.") var microbatch: Int? = nil

  // Model hyperparameters
  @Option(name: .long, help: "State count.") var stateCount: Int = 8
  @Option(name: .long, help: "MLP hidden size.") var hiddenSize: Int = 16
  @Option(name: .long, help: "Number of cells.") var cellCount: Int = 1024
  @Option(name: .long, help: "Activations per cell.") var actPerCell: Int = 16
  @Option(name: .long, help: "Normalization mode.") var normalization: Cell.Normalization = .none

  // CMA-ES
  @Option(name: .long, help: "Initial standard deviation for search.") var initStepSize: Float =
    0.001
  @Option(name: .long, help: "Fraction of population to survive.") var recombinationFrac: Float =
    0.5
  @Option(name: .long, help: "Maximum population size per forward.") var populationMaxBatch: Int? =
    nil
  @Option(name: .long, help: "Loss evaluations between eigendecompositions.") var samplesPerEig:
    Int? = nil
  @Option(name: .long, help: "Population size.") var population: Int? = nil

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

      func unflattenParamBatch(_ batch: Tensor) -> [String: Tensor] {
        var offset = 0
        var result = [String: Tensor]()
        for (name, param) in cell.parameters {
          let shape = param.data!.shape
          let size = shape.reduce(1, *)
          result[name] = batch[..., offset..<(offset + size)].reshape(
            Array([batch.shape[0]] + shape)
          )
          offset += size
        }
        return result
      }

      let opt = CMA(
        config: .init(stepSize: initStepSize, population: population, samplesPerEig: samplesPerEig),
        initialValue: Tensor(concat: cell.parameters.map { $0.1.data!.flatten() })
      )

      var dataIt = try await MNISTIterator(batchSize: batchSize)

      var step: Int = 0

      if FileManager.default.fileExists(atPath: outputPath) {
        print("loading from checkpoint: \(outputPath) ...")
        let data = try Data(contentsOf: URL(fileURLWithPath: outputPath))
        let decoder = PropertyListDecoder()
        let state = try decoder.decode(State.self, from: data)
        try cell.loadState(state.model)
        if let optState = state.cma { try opt.loadState(optState) }
        if let dataState = state.data { dataIt.state = dataState }
        step = state.step ?? 0
        opt.mean = Tensor(concat: cell.parameters.map { $0.1.data!.flatten() })
      }

      while true {
        let (allInputs, allLabelIndices, allLabels) = dataIt.rollout(count: examplesPerRollout)!

        func computeMetrics(params: Tensor? = nil) async throws -> Metrics {
          var allMetrics = [Metrics]()
          let mbSize = microbatch ?? batchSize
          for i in stride(from: 0, to: batchSize, by: mbSize) {
            let curMbSize = min(batchSize - i, mbSize)
            let mbScale = Float(curMbSize) / Float(batchSize)
            let graph = try await Graph.random(
              batchSize: curMbSize,
              cellCount: cellCount,
              actPerCell: actPerCell,
              inCount: 28 * 28,
              outCount: 10
            )

            let count = if let p = params { p.shape[0] } else { 1 }
            let rollouts = Tensor.withGrad(enabled: false) {
              Rollout.rollout(
                inferSteps: inferSteps,
                updateSteps: updateSteps,
                inputs: allInputs.map { $0[i..<(i + curMbSize)].repeating(axis: 0, count: count) },
                targets: allLabels.map { $0[i..<(i + curMbSize)].repeating(axis: 0, count: count) },
                cell: cell,
                graph: graph.repeated(count: count),
                params: params == nil ? nil : unflattenParamBatch(params!)
              )
            }

            func means(_ x: Tensor) -> Tensor {
              let x = x.reshape([count, x.shape[0] / count] + x.shape[1...])
              return x.flatten(startAxis: 1).mean(axis: 1)
            }

            let subLabelIndices = allLabelIndices.map {
              $0[i..<(i + curMbSize)].repeating(axis: 0, count: count)
            }
            let losses = zip(subLabelIndices, rollouts.outputs).map { (labelIdxs, logits) in
              means(logits.logSoftmax(axis: -1).gather(axis: 1, indices: labelIdxs[..., NewAxis()]))
            }
            let accs = zip(subLabelIndices, rollouts.outputs).map { (labelIdxs, logits) in
              means((labelIdxs == logits.argmax(axis: 1)).cast(.float32))
            }
            let meanLoss = -Tensor(stack: losses).mean(axis: 0) * mbScale
            let meanAcc = Tensor(stack: accs).mean(axis: 0) * mbScale

            try await meanLoss.wait()
            try await meanAcc.wait()

            allMetrics.append([.loss: meanLoss, .accuracy: meanAcc])
          }
          let concatMetrics = Metrics(
            uniqueKeysWithValues: allMetrics[0].keys.map { k in
              (k, Tensor(stack: allMetrics.map { $0[k]! }).sum(axis: 0))
            }
          )
          return concatMetrics
        }

        let population = opt.sample()
        let populationBatch =
          populationMaxBatch == nil
          ? population.shape[0] : min(population.shape[0], populationMaxBatch!)
        var allLosses = [Tensor]()
        for i in stride(from: 0, to: population.shape[0], by: populationBatch) {
          allLosses.append(
            try await computeMetrics(
              params: population[i..<min(population.shape[0], i + populationBatch)]
            )[.loss]!
          )
        }
        opt.update(samples: population, scores: Tensor(concat: allLosses))

        let newParams = unflattenParamBatch(opt.mean[NewAxis()])
        for (name, var param) in cell.parameters { param.data! = newParams[name]![0] }

        var metrics = try await computeMetrics()
        metrics[.named("step_size")] = opt.sigma
        metrics[.named("cov_cond")] = opt.covarianceCondition
        step += 1
        print("step \(step): \(try await metrics.format())")

        if step % saveInterval == 0 {
          print("saving after \(step) steps...")
          let state = State(
            model: try await cell.state(),
            step: step,
            data: dataIt.state,
            cma: try await opt.state()
          )
          let stateData = try PropertyListEncoder().encode(state)
          try stateData.write(to: URL(filePath: outputPath), options: .atomic)
        }
      }
    } catch { print("FATAL ERROR: \(error)") }
  }
}
