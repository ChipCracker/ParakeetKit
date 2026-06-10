//
//  BenchEnv.swift
//  Shared plumbing for the engine benchmarks: model path via env, bundled
//  jfk.wav, reference transcript, result JSON export.
//
//  The whole file is iOS-only — on macOS the target compiles empty so that
//  plain `swift test` keeps working.
//
#if os(iOS)
import Foundation
import XCTest
import ParakeetKit

enum BenchEnv {
    /// GGUF path, injected by benchmark.sh as TEST_RUNNER_PARAKEET_BENCH_MODEL
    /// (xcodebuild strips the prefix before handing it to the test process).
    static var modelPath: String? {
        ProcessInfo.processInfo.environment["PARAKEET_BENCH_MODEL"]
    }

    /// Resolves the benchmark model, in order:
    /// 1. PARAKEET_BENCH_MODEL path — simulator runs read it from the host.
    /// 2. A model already in the app container (previous on-device run).
    /// 3. PARAKEET_BENCH_DOWNLOAD=1 — fetch q4_K (466 MB) via ModelDownloader;
    ///    physical devices can't see host paths, so they download once and
    ///    cache in Application Support.
    static func resolveModelOrSkip() async throws -> String {
        if let path = modelPath, FileManager.default.fileExists(atPath: path) {
            return path
        }
        let downloader = ModelDownloader()
        let spec = ParakeetModelCatalog.q4_K
        if case .ready(let url) = downloader.state(for: spec) {
            return url.path
        }
        if ProcessInfo.processInfo.environment["PARAKEET_BENCH_DOWNLOAD"] != nil {
            print("[bench] downloading \(spec.fileName) (~466 MB) to the device …")
            final class ProgressGate: @unchecked Sendable {
                private let lock = NSLock()
                private var last = -1
                func tenth(_ p: Double) -> Int? {
                    lock.lock(); defer { lock.unlock() }
                    let t = Int(p * 10)
                    guard t > last else { return nil }
                    last = t
                    return t
                }
            }
            let gate = ProgressGate()
            let url = try await downloader.download(spec) { progress in
                if let t = gate.tenth(progress) { print("[bench] download \(t * 10)%") }
            }
            return url.path
        }
        throw XCTSkip("no model: set PARAKEET_BENCH_MODEL (simulator) or "
                      + "PARAKEET_BENCH_DOWNLOAD=1 (device) — see scripts/benchmark.sh")
    }

    /// Resource bundle for jfk.wav: SPM generates Bundle.module; the
    /// xcodegen-hosted device target (project.yml) uses the test bundle itself.
    private final class BundleMarker {}
    static var resourceBundle: Bundle {
        #if SWIFT_PACKAGE
        return Bundle.module
        #else
        return Bundle(for: BundleMarker.self)
        #endif
    }

    /// The bundled 11 s JFK sample (16 kHz mono).
    static func loadJFK() throws -> [Float] {
        guard let url = resourceBundle.url(forResource: "jfk", withExtension: "wav") else {
            throw XCTSkip("bundled jfk.wav missing")
        }
        return try AudioFileLoader.loadSamples(url: url)
    }

    /// Concatenates `count` copies, separated by `gapSeconds` of silence.
    static func chain(_ samples: [Float], count: Int, gapSeconds: Double) -> [Float] {
        let gap = [Float](repeating: 0, count: Int(gapSeconds * 16_000))
        var out: [Float] = []
        out.reserveCapacity(samples.count * count + gap.count * max(0, count - 1))
        for i in 0..<count {
            if i > 0 { out.append(contentsOf: gap) }
            out.append(contentsOf: samples)
        }
        return out
    }

    static let jfkReference =
        "And so, my fellow Americans: ask not what your country can do for you — "
        + "ask what you can do for your country."

    static func reference(times: Int) -> String {
        Array(repeating: jfkReference, count: times).joined(separator: " ")
    }
}

/// One engine-benchmark measurement (single shot, long audio variant, E2E run).
struct EngineBenchResult: Codable {
    let name: String
    let wer: Double
    let rtf: Double
    let encoderRuns: Int
    let decoderSteps: Int
    let audioSeconds: Double
    let processingSeconds: Double
    // E2E-only:
    var transcribeCalls: Int? = nil
    var transcribedSecondsTotal: Double? = nil
    var totalProcessingSeconds: Double? = nil   // ALL runs (previews+commits+final)
    var committedWER: Double? = nil
    var finalizedWER: Double? = nil

    init(name: String, wer: Double, transcript: ParakeetTranscript) {
        self.name = name
        self.wer = wer
        self.rtf = transcript.rtf
        self.encoderRuns = transcript.encoderRuns
        self.decoderSteps = transcript.decoderSteps
        self.audioSeconds = transcript.audioSeconds
        self.processingSeconds = transcript.processingSeconds
    }
}

enum BenchJSON {
    /// Pretty JSON to $PARAKEET_BENCH_OUT/<name>.json (if set) + console echo.
    /// Physical devices can't write to host paths — the single-line
    /// "[bench-json] <name> <json>" marker lets benchmark.sh recover the
    /// results from the xcodebuild log instead.
    static func write<T: Encodable>(_ value: T, name: String) {
        let compact = JSONEncoder()
        compact.outputFormatting = [.sortedKeys]
        if let data = try? compact.encode(value), let line = String(data: data, encoding: .utf8) {
            print("[bench-json] \(name) \(line)")
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        guard let data = try? encoder.encode(value),
              let json = String(data: data, encoding: .utf8) else { return }
        print("[bench] \(name):\n\(json)")
        guard let dir = ProcessInfo.processInfo.environment["PARAKEET_BENCH_OUT"] else { return }
        let dirURL = URL(fileURLWithPath: dir, isDirectory: true)
        try? FileManager.default.createDirectory(at: dirURL, withIntermediateDirectories: true)
        try? data.write(to: dirURL.appendingPathComponent("\(name).json"))
    }
}

/// Counts every transcribe call the streaming session makes (previews and
/// commits — `session.stats` only covers commits) plus pure inference time.
actor BenchCallRecorder {
    private(set) var calls = 0
    private(set) var samplesTotal = 0
    private(set) var processingSeconds: Double = 0

    func record(sampleCount: Int, processing: Double) {
        calls += 1
        samplesTotal += sampleCount
        processingSeconds += processing
    }

    var audioSeconds: Double { Double(samplesTotal) / 16_000 }
}
#endif
