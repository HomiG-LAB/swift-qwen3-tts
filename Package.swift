// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "swift-qwen3-tts",
    platforms: [
        .macOS("15.0"),
        .iOS("18.0"),
    ],
    products: [
        // Core Qwen3 TTS library
        .library(
            name: "Qwen3TTS",
            targets: ["Qwen3TTS"]
        ),
        // Command line demo tool
        .executable(
            name: "Qwen3TTSDemo",
            targets: ["Qwen3TTSDemo"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.31.0"),
        // Pinned to a main-branch commit because the latest tagged release
        // (2.29.1) pins mlx-swift 0.29.1..<0.30.0, which conflicts with
        // downstream projects on mlx-swift 0.31+. Main already tracks 0.31.3.
        .package(url: "https://github.com/ml-explore/mlx-swift-examples/", revision: "357c97fbd39abe600704b889dd114c208b0ed915"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.0.0"),
    ],
    targets: [
        // MARK: - Core Library
        .target(
            name: "Qwen3TTS",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-examples"),
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "Sources/Qwen3TTS",
            swiftSettings: [
                .unsafeFlags(["-Xfrontend", "-warn-concurrency"], .when(configuration: .debug))
            ]
        ),

        // MARK: - CLI Demo
        .executableTarget(
            name: "Qwen3TTSDemo",
            dependencies: [
                "Qwen3TTS",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Sources/Qwen3TTSDemo"
        ),

        // MARK: - Tests
        .testTarget(
            name: "Qwen3TTSTests",
            dependencies: [
                "Qwen3TTS",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Tests/Qwen3TTSTests"
        ),
    ]
)
