import ArgumentParser
import Honeycrisp
import MNIST

public struct MNISTIterator: Sequence, IteratorProtocol {
  public enum Source: String, ExpressibleByArgument, CaseIterable, Sendable {
    case mnist
    case fashionMNIST
  }

  public struct State: Codable {
    public let offset: Int
    public let permutation: [Int]
  }

  let images: [MNISTDataset.Image]
  let batchSize: Int
  var permutation: [Int]
  var offset = 0

  public var state: State {
    get { State(offset: offset, permutation: permutation) }
    set {
      permutation = newValue.permutation
      offset = newValue.offset
    }
  }

  public init(images: [MNISTDataset.Image], batchSize: Int) {
    self.images = images
    self.batchSize = batchSize
    permutation = Array(0..<images.count).shuffled()
  }

  public init(batchSize: Int, source: Source = .mnist) async throws {
    let mnistSource: MNISTDataset.Source =
      switch source {
      case .mnist: .mnist
      case .fashionMNIST: .fashionMNIST
      }
    self.init(
      images: try await MNISTDataset.download(toDir: "\(source.rawValue)_data", source: mnistSource)
        .train,
      batchSize: batchSize
    )
  }

  public mutating func next() -> (Tensor, Tensor)? {
    var inputData = [Float]()
    var outputLabels = [Int]()
    for _ in 0..<batchSize {
      if offset >= images.count {
        permutation = Array(0..<images.count).shuffled()
        offset = offset % images.count
      }
      let img = images[permutation[offset % images.count]]
      for pixel in img.pixels { inputData.append(Float(pixel) / 255) }
      outputLabels.append(img.label)
      offset += 1
    }
    let probs = Tensor(data: inputData, shape: [batchSize, 28 * 28], dtype: .float32)
    let pixels = (probs > Tensor(randLike: probs)).cast(.float32) * 2 - 1
    return (pixels, Tensor(data: outputLabels, dtype: .int64))
  }

  public mutating func rollout(count: Int) -> (
    inputs: [Tensor], labelIndices: [Tensor], labels: [Tensor]
  )? {
    var allInputs = [Tensor]()
    var allLabelIndices = [Tensor]()
    var allLabels = [Tensor]()
    for _ in 0..<count {
      guard let (inputs, labels) = next() else { return nil }
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
    return (inputs: allInputs, labelIndices: allLabelIndices, labels: allLabels)
  }
}
