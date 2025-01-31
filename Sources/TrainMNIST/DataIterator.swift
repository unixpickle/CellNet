import Honeycrisp
import MNIST

struct DataIterator: Sequence, IteratorProtocol {
  struct State: Codable {
    let offset: Int
    let permutation: [Int]
  }

  let images: [MNISTDataset.Image]
  let batchSize: Int
  var permutation: [Int]
  var offset = 0

  var state: State {
    get { State(offset: offset, permutation: permutation) }
    set {
      permutation = newValue.permutation
      offset = newValue.offset
    }
  }

  init(images: [MNISTDataset.Image], batchSize: Int) {
    self.images = images
    self.batchSize = batchSize
    permutation = Array(0..<images.count).shuffled()
  }

  mutating func next() -> (Tensor, Tensor)? {
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
}
