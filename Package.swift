// swift-tools-version:6.0
import PackageDescription
import Foundation

// MARK: - CParakeet binary target (env-switch: local dev vs. remote release)
//
// Parakeet.xcframework is a STATIC-library xcframework (libparakeet-combined.a +
// Headers, with an injected module.modulemap exposing the Clang module
// `CParakeet`). It is iOS-only (ios-arm64 device + ios-arm64-simulator), so the
// package can be built/run only for iOS — `swift test` on macOS covers the pure
// `ParakeetCore` target.

let remoteURL = "https://github.com/ChipCracker/ParakeetKit/releases/download/parakeet-1/Parakeet.xcframework.zip"
let remoteChecksum = "475d9eec37f42b629bbd8fafe63e0d51bbcdd62257ea6d33ad285e479db61d10"

let localXCFrameworkPath = "Frameworks/Parakeet.xcframework"
let hasLocal = FileManager.default.fileExists(atPath: localXCFrameworkPath)
let forceLocal = ProcessInfo.processInfo.environment["PARAKEETKIT_LOCAL_XCFRAMEWORK"] != nil
let useLocal = forceLocal || hasLocal || remoteChecksum.isEmpty

let parakeetBinaryTarget: Target = useLocal
    ? .binaryTarget(name: "CParakeet", path: localXCFrameworkPath)
    : .binaryTarget(name: "CParakeet", url: remoteURL, checksum: remoteChecksum)

let package = Package(
    name: "ParakeetKit",
    platforms: [
        .iOS(.v16),     // ParakeetKit (binary) slices: ios-arm64 + ios-arm64-simulator only
        .macOS(.v13),   // only ParakeetCore (pure logic) is buildable on macOS
    ],
    products: [
        .library(name: "ParakeetKit", targets: ["ParakeetKit"]),   // consumers import this
        .library(name: "ParakeetCore", targets: ["ParakeetCore"]), // pure logic, no binary
    ],
    targets: [
        // Prebuilt static-library xcframework, exposed as Clang module `CParakeet`.
        parakeetBinaryTarget,

        // Pure Swift logic — Foundation only, NO CParakeet import → testable on macOS.
        .target(
            name: "ParakeetCore",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),

        // Binary + AVFoundation layer (iOS only). Re-exports ParakeetCore.
        // On macOS the sources compile out (#if os(iOS)) so the package's
        // ParakeetCore tests stay runnable via host `swift test`.
        .target(
            name: "ParakeetKit",
            dependencies: [
                "ParakeetCore",
                .target(name: "CParakeet", condition: .when(platforms: [.iOS])),
            ],
            resources: [
                // Bundled streaming VAD model (~2.2 MB) → Bundle.module.
                .copy("Resources/firered-stream-vad.gguf"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)],
            linkerSettings: [
                // The static archive bundles ggml/ggml-metal → link these explicitly
                // (a non-framework modulemap can't autolink them).
                .linkedLibrary("c++", .when(platforms: [.iOS])),
                .linkedFramework("Metal", .when(platforms: [.iOS])),
                .linkedFramework("Accelerate", .when(platforms: [.iOS])),
                .linkedFramework("AVFoundation", .when(platforms: [.iOS])),
            ]
        ),

        // Pure-logic tests (streaming state machine + model catalog) — run on macOS.
        .testTarget(
            name: "ParakeetCoreTests",
            dependencies: ["ParakeetCore"]
        ),

        // Engine benchmarks (real inference) — iOS only; every source file is
        // wrapped in #if os(iOS) so `swift test` on macOS compiles them empty.
        // Run via scripts/benchmark.sh (xcodebuild against an iOS simulator,
        // model path injected via TEST_RUNNER_PARAKEET_BENCH_MODEL).
        .testTarget(
            name: "ParakeetKitBenchmarks",
            dependencies: ["ParakeetKit", "ParakeetCore"],
            resources: [
                .copy("Resources/jfk.wav"),
                // Two qwen3-tts CustomVoice speakers (ryan/serena) for
                // deterministic diarization tests.
                .copy("Resources/voice-ryan.wav"),
                .copy("Resources/voice-serena.wav"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
