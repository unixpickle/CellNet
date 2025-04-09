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
    let fdState: FiniteDiffs.State?
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

  // Other optimizer hyperparameters
  @Option(name: .long, help: "Maximum gradient RMS before clipping.") var maxGradRMS: Float = 100.0
  @Option(
    name: .long,
    help:
      "If specified, step along the gradient (rescaled to this norm) and penalize gradients that change too much"
  ) var sharpnessClipDelta: Float? = nil
  @Option(name: .long, help: "Threshold under which to completely clip gradients")
  var sharpnessClipThreshold: Float = 0.1
  @Option(name: .long, help: "Parameter decay rate when sharp gradients are clipped")
  var sharpnessClipDecay: Float = 0.995
  @Option(
    name: .long,
    help: "If the loss gets this much worse during sharpness clipping, clip the gradient to zero"
  ) var sharpnessLossDelta: Float = 0.01

  // Evolutions strategies
  @Option(name: .long, help: "If specified, use Evolution Strategies with this epsilon.")
  var esEpsilon: Float? = nil
  @Option(name: .long, help: "The ES population size.") var esPopulation: Int = 1
  @Option(name: .long, help: "The ES batch size.") var esBatchSize: Int = 1

  // Finite differences hyperparameters
  @Option(name: .long, help: "If specified, use finite differences.") var fdEpsilon: Float? = nil
  @Option(name: .long, help: "Number of axes to search in finite differences.") var fdAxes: Int = 16
  @Option(name: .long, help: "Finite differences minibatch size.") var fdBatchSize: Int = 1

  // Saving
  @Option(name: .shortAndLong, help: "Output path.") var outputPath: String = "state.plist"
  @Option(name: .long, help: "Save interval.") var saveInterval: Int = 100

  mutating func run() async {
    print("Command:", CommandLine.arguments.joined(separator: " "))

    if fdEpsilon != nil && esEpsilon != nil {
      print("ERROR: cannot mix ES and finite differences")
      return
    }

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
      let fd: FiniteDiffs? =
        if let eps = fdEpsilon { FiniteDiffs(model: cell, eps: eps, evalCount: fdAxes) } else {
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
        if let fdState = state.fdState, let fd = fd { fd.loadState(fdState) }
        step = state.step ?? 0
      }

      while true {
        let (allInputs, allLabelIndices, allLabels) = dataIt.rollout(count: examplesPerRollout)!

        let esNoise = es?.sampleNoises()
        let fdAxes = fd?.sampleAxes()

        func computeGradients() async throws -> Metrics {
          var allMetrics = [Metrics]()
          let actGradClipper = ActGradClipper(maxRMS: maxGradRMS)
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
                params: params,
                clipper: actGradClipper
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
              } else if let fd = fd, let fdAxes = fdAxes {
                try await fd.forwardBackward(sample: fdAxes, batchSize: fdBatchSize) { x, y in
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
          var metrics = Metrics.sum(allMetrics)
          metrics += actGradClipper.metrics
          return metrics
        }

        var metrics = try await computeGradients()
        metrics += try await sharpnessClip(oldMetrics: metrics, cell: cell, opt: opt) {
          if microbatch != nil { try await waitForGrads() }
          opt.clearGrads()
          let result = try await computeGradients()
          if microbatch != nil { try await waitForGrads() }
          return result
        }

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
            clipper: clipper.state,
            fdState: fd?.state()
          )
          let stateData = try PropertyListEncoder().encode(state)
          try stateData.write(to: URL(filePath: outputPath), options: .atomic)
        }
      }
    } catch { print("FATAL ERROR: \(error)") }
  }

  @recordCaller private func _sharpnessClip(
    oldMetrics: Metrics,
    cell: TrainableProto,
    opt: Muon,
    gradCallback: () async throws -> Metrics
  ) async throws -> Metrics {
    guard let delta = sharpnessClipDelta else { return [:] }
    var oldParams = [Tensor]()
    var oldGrads = [Tensor]()
    var sqSum = Tensor(data: [0.0], shape: [])
    for (_, param) in cell.parameters {
      if let data = param.data, let grad = param.grad {
        oldParams.append(data)
        oldGrads.append(grad)
        sqSum = sqSum + grad.pow(2).sum()
      }
    }
    let scale = (delta / (sqSum.sqrt() + 1e-8))
    for (_, var param) in cell.parameters {
      if let grad = param.grad, let data = param.data { param.data = data - grad * scale }
    }

    let newMetrics = try await gradCallback()

    var newSqSum = Tensor(data: [0.0], shape: [])
    var gradDot = newSqSum
    for (_, var param) in cell.parameters {
      if let newGrad = param.grad {
        param.grad = oldGrads.removeFirst()
        newSqSum = newSqSum + newGrad.pow(2).sum()
        gradDot = gradDot + (newGrad * param.grad!).sum()
        param.data = oldParams.removeFirst()
      }
    }

    let correlation = (gradDot / (sqSum.sqrt() * newSqSum.sqrt() + 1e-8))
    let lossDelta = (oldMetrics[.loss]! - newMetrics[.loss]!)
    let correlationItem = try await correlation.item()
    let lossDeltaItem = try await lossDelta.item()
    if lossDeltaItem < -sharpnessLossDelta || correlationItem < sharpnessClipThreshold {
      for (_, var param) in cell.parameters {
        param.grad = nil
        if let data = param.data { param.data = data * sharpnessClipDecay }
      }
      opt.moments = [:]
    } else {
      for (_, var param) in cell.parameters {
        if let grad = param.grad { param.grad = grad * correlation }
      }
    }
    return [.named("sharpness_scale"): correlation, .named("loss_delta"): lossDelta]
  }
}
