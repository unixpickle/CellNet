// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
  name: "SynthClone",
  platforms: [.macOS(.v13)],
  products: [
    .library(name: "CellNet", targets: ["CellNet"]),
    .library(name: "DataUtils", targets: ["DataUtils"]),
  ],
  dependencies: [
    .package(url: "https://github.com/unixpickle/honeycrisp", from: "0.0.29"),
    .package(url: "https://github.com/unixpickle/swift-cma", from: "0.1.0"),
    .package(url: "https://github.com/apple/swift-argument-parser", from: "1.3.0"),
    .package(url: "https://github.com/unixpickle/honeycrisp-examples.git", from: "0.0.2"),
  ],
  targets: [
    .target(
      name: "CellNet",
      dependencies: [
        .product(name: "ArgumentParser", package: "swift-argument-parser"),
        .product(name: "Honeycrisp", package: "honeycrisp"),
        .product(name: "HCBacktrace", package: "honeycrisp"),
      ]
    ),
    .target(
      name: "DataUtils",
      dependencies: [
        .product(name: "MNIST", package: "honeycrisp-examples"),
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
        "CellNet", "DataUtils", .product(name: "ArgumentParser", package: "swift-argument-parser"),
        .product(name: "Honeycrisp", package: "honeycrisp"),
      ]
    ),
    .executableTarget(
      name: "TrainMNISTWithCMA",
      dependencies: [
        "CellNet", "DataUtils", .product(name: "ArgumentParser", package: "swift-argument-parser"),
        .product(name: "Honeycrisp", package: "honeycrisp"),
        .product(name: "CMA", package: "swift-cma"),
      ]
    ),
  ]
)
