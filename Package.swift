// swift-tools-version: 6.0
import PackageDescription
import Foundation

// caix — native Apple Core AI inference server for Apple silicon (BETA).
//
// Apple's Core AI runtime (the `CoreAILM` product from apple/coreai-models plus the `CoreAI`
// system framework) is currently in BETA and requires a recent macOS / Xcode beta. To keep the
// package building on stock toolchains, the runtime is opt-in: set COREAI_RUNTIME=1 to link the
// Apple Swift runtime + tokenizer and raise the deployment target. With the flag unset the package
// compiles standalone (dashboard + API surface build, inference returns 503). See README.md.
let enableCoreAIRuntime = ProcessInfo.processInfo.environment["COREAI_RUNTIME"] == "1"

// Hummingbird powers the HTTP layer — added unconditionally so `serve` builds on stock toolchains.
var packageDependencies: [Package.Dependency] = [
    .package(url: "https://github.com/hummingbird-project/hummingbird.git", from: "2.0.0")
]
var runtimeDependencies: [Target.Dependency] = []
var runtimeSwiftSettings: [SwiftSetting] = []

if enableCoreAIRuntime {
    packageDependencies.append(
        .package(
            url: "https://github.com/apple/coreai-models.git",
            revision: "34f0db331dd69d0b295d5f69b3edce7347115e43")
    )
    packageDependencies.append(
        .package(url: "https://github.com/huggingface/swift-transformers", exact: "1.3.3")
    )
    runtimeDependencies.append(.product(name: "CoreAILM", package: "coreai-models"))
    runtimeDependencies.append(.product(name: "Transformers", package: "swift-transformers"))
    runtimeSwiftSettings.append(.define("COREAI_RUNTIME"))
}

let platforms: [SupportedPlatform] = enableCoreAIRuntime ? [.macOS("27.0")] : [.macOS("14.0")]

let package = Package(
    name: "caix",
    platforms: platforms,
    products: [
        .executable(name: "caix", targets: ["PipelineCLI"]),
        .library(name: "MachineStats", targets: ["MachineStats"]),
        .library(name: "PipelineRuntime", targets: ["PipelineRuntime"]),
    ],
    dependencies: packageDependencies,
    targets: [
        .target(name: "MachineStats"),
        .target(
            name: "PipelineRuntime",
            dependencies: runtimeDependencies,
            swiftSettings: runtimeSwiftSettings
        ),
        .target(
            name: "CoreAIServer",
            dependencies: [
                "MachineStats",
                "PipelineRuntime",
                .product(name: "Hummingbird", package: "hummingbird"),
            ],
            swiftSettings: runtimeSwiftSettings
        ),
        .executableTarget(
            name: "PipelineCLI",
            dependencies: ["MachineStats", "PipelineRuntime", "CoreAIServer"],
            swiftSettings: runtimeSwiftSettings
        ),
        .testTarget(name: "MachineStatsTests", dependencies: ["MachineStats"]),
        .testTarget(
            name: "PipelineRuntimeTests",
            dependencies: ["PipelineRuntime"],
            swiftSettings: runtimeSwiftSettings
        ),
        .testTarget(
            name: "CoreAIServerTests",
            dependencies: ["CoreAIServer"],
            swiftSettings: runtimeSwiftSettings
        ),
    ]
)
