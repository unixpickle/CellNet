import Honeycrisp
import MNIST

struct DataIterator: Sequence, IteratorProtocol {
  let images: [MNISTDataset.Image]
  let batchSize: Int
  var offset = 0

  mutating func next() -> (Tensor, Tensor)? {
    var inputData = [Float]()
    var outputLabels = [Int]()
    for _ in 0..<batchSize {
      let img = images[offset % images.count]
      for pixel in img.pixels { inputData.append(Float(pixel) / 255) }
      outputLabels.append(img.label)
      offset += 1
    }
    let probs = Tensor(data: inputData, shape: [batchSize, 28 * 28], dtype: .float32)
    let pixels = (probs > Tensor(randLike: probs)).cast(.float32)
    return (pixels, Tensor(data: outputLabels, dtype: .int64))
  }
}
