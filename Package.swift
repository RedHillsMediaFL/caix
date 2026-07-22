// swift-tools-version: 6.0
import PackageDescription
import Foundation

// caix — native Apple Core AI inference server for Apple silicon (BETA).
//
// Apple's Core AI runtime is currently in BETA and requires a recent macOS / Xcode beta. There are
// two explicit opt-in build modes:
//
// - COREAI_RUNTIME=1 links CoreAILM, CoreAI, and swift-transformers.
// - COREAI_DIRECT_RUNTIME=1 links CoreAI and swift-transformers without CoreAILM. This keeps
//   CAIX's direct executors usable when the installed OS and CoreAILM's FoundationModels ABI do
//   not match. If both are set, direct mode deliberately wins.
//
// With both flags unset the package compiles standalone (dashboard + API surface build, inference
// returns 503). See README.md.
let enableDirectCoreAIRuntime =
    ProcessInfo.processInfo.environment["COREAI_DIRECT_RUNTIME"] == "1"
let enableFullCoreAIRuntime =
    ProcessInfo.processInfo.environment["COREAI_RUNTIME"] == "1" && !enableDirectCoreAIRuntime
let enableCoreAIRuntime = enableFullCoreAIRuntime || enableDirectCoreAIRuntime

// Hummingbird powers the HTTP layer — added unconditionally so `serve` builds on stock toolchains.
var packageDependencies: [Package.Dependency] = [
    .package(url: "https://github.com/hummingbird-project/hummingbird.git", from: "2.0.0")
]
var runtimeDependencies: [Target.Dependency] = []
var runtimeSwiftSettings: [SwiftSetting] = []

if enableFullCoreAIRuntime {
    packageDependencies.append(
        .package(
            url: "https://github.com/apple/coreai-models.git",
            branch: "main")
    )
    runtimeDependencies.append(.product(name: "CoreAILM", package: "coreai-models"))
}

if enableCoreAIRuntime {
    packageDependencies.append(
        .package(url: "https://github.com/huggingface/swift-transformers", exact: "1.3.3")
    )
    // PipelineRuntime renders the SHA-locked July Gemma 4 template directly so rich tool and
    // reasoning objects can be validated against the exact rendered prompt before tokenization.
    packageDependencies.append(
        .package(url: "https://github.com/huggingface/swift-jinja.git", exact: "2.4.2")
    )
    runtimeDependencies.append(.product(name: "Transformers", package: "swift-transformers"))
    runtimeDependencies.append(.product(name: "Jinja", package: "swift-jinja"))
    runtimeSwiftSettings.append(.define("COREAI_RUNTIME"))
}

if enableDirectCoreAIRuntime {
    runtimeSwiftSettings.append(.define("COREAI_DIRECT_RUNTIME"))
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
            swiftSettings: runtimeSwiftSettings,
            linkerSettings: [
                .linkedFramework("AudioToolbox"),
            ]
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
