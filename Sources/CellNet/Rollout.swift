import Honeycrisp

public struct Rollout {
  public let inferSteps: Int
  public let updateSteps: Int

  public let inputs: [Tensor]
  public let outputs: [Tensor]
}
