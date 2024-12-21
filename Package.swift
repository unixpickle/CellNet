// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
  name: "SynthClone",
  platforms: [.macOS(.v13)],
  products: [.library(name: "CellNet", targets: ["CellNet"])],
  dependencies: [
    .package(url: "https://github.com/unixpickle/honeycrisp", from: "0.0.16"),
    .package(url: "https://github.com/apple/swift-argument-parser", from: "1.3.0"),
    .package(url: "https://github.com/unixpickle/honeycrisp-examples.git", from: "0.0.2"),
  ],
  targets: [
    .target(
      name: "CellNet",
      dependencies: [
        .product(name: "Honeycrisp", package: "honeycrisp"),
        .product(name: "HCBacktrace", package: "honeycrisp"),
      ]
    ),
    .executableTarget(
      name: "TrainLogic",
      dependencies: [
        "CellNet", .product(name: "ArgumentParser", package: "swift-argument-parser"),
        .product(name: "Honeycrisp", package: "honeycrisp"),
      ]
    ),
    .executableTarget(
      name: "TrainMNIST",
      dependencies: [
        "CellNet", .product(name: "ArgumentParser", package: "swift-argument-parser"),
        .product(name: "MNIST", package: "honeycrisp-examples"),
        .product(name: "Honeycrisp", package: "honeycrisp"),
      ]
    ),
  ]
)
