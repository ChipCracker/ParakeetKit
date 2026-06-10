//
//  EngineBenchmarkTests.swift
//  Real-inference benchmarks against the bundled jfk.wav. Skipped unless a
//  model path is injected (scripts/benchmark.sh). Simulator note: GPU is off
//  (`preferredUseGPU`), so timings are CPU-path numbers — quality asserts
//  (WER / text parity) are the hard criteria, times are recorded for trends.
//
#if os(iOS)
import XCTest
import ParakeetKit

final class EngineBenchmarkTests: XCTestCase {

    /// Single-shot and long-audio share one engine: loading the 1.26 GB F16
    /// weights once per test pushes the test process into iPadOS memory
    /// limits (the 5th sequential load fails with posix_memalign). Kept alive
    /// for the suite; XCTSkip from model resolution propagates through.
    private static var sharedEngineTask: Task<ParakeetEngine, Error>?
    private static func sharedEngine() async throws -> ParakeetEngine {
        if let task = sharedEngineTask { return try await task.value }
        let task = Task<ParakeetEngine, Error> {
            let model = try await BenchEnv.resolveModelOrSkip()
            return try await ParakeetEngine.make(modelPath: model,
                                                 useGPU: ParakeetEngine.preferredUseGPU)
        }
        sharedEngineTask = task
        return try await task.value
    }

    func testSingleShotJFK() async throws {
        let jfk = try BenchEnv.loadJFK()
        let engine = try await Self.sharedEngine()

        let result = await engine.transcribe(jfk)
        let wer = WordErrorRate.wer(reference: BenchEnv.jfkReference, hypothesis: result.text)

        let bench = EngineBenchResult(name: "single-shot-jfk", wer: wer, transcript: result)
        BenchJSON.write(bench, name: "single-shot")

        XCTAssertLessThanOrEqual(wer, 0.10, "transcript degraded: \(result.text)")
        XCTAssertEqual(result.encoderRuns, 1)
    }

    /// Flash attention must not change the transcript (upstream: bit-identical
    /// on Metal). Engines are loaded sequentially — 2× 466 MB not at once.
    /// Simulator timings are noisy CPU numbers (±30 % between runs), so each
    /// variant is measured 3× and the median recorded; the hard assert is parity.
    func testFlashAttentionParity() async throws {
        let model = try await BenchEnv.resolveModelOrSkip()
        let jfk = try BenchEnv.loadJFK()

        func median3(_ engine: ParakeetEngine) async -> (text: String, seconds: Double) {
            var text = ""
            var times: [Double] = []
            for _ in 0..<3 {
                let r = await engine.transcribe(jfk)
                text = r.text
                times.append(r.processingSeconds)
            }
            return (text, times.sorted()[1])
        }

        var noFlashText = ""
        var noFlashSeconds = 0.0
        do {
            let engine = try await ParakeetEngine.make(modelPath: model,
                                                       useGPU: ParakeetEngine.preferredUseGPU,
                                                       useFlashAttention: false)
            (noFlashText, noFlashSeconds) = await median3(engine)
        }

        let engine = try await ParakeetEngine.make(modelPath: model,
                                                   useGPU: ParakeetEngine.preferredUseGPU,
                                                   useFlashAttention: true)
        let (flashText, flashSeconds) = await median3(engine)

        let wer = WordErrorRate.wer(reference: BenchEnv.jfkReference, hypothesis: flashText)
        let bench = ["name": "flash-attention-parity", "wer": "\(wer)",
                     "flashMedianSeconds": "\(flashSeconds)",
                     "noFlashMedianSeconds": "\(noFlashSeconds)"]
        BenchJSON.write(bench, name: "flash-parity")

        XCTAssertEqual(WordErrorRate.normalize(noFlashText), WordErrorRate.normalize(flashText),
                       "flash attention changed the transcript")
        XCTAssertLessThanOrEqual(wer, 0.10)
        print("[bench] flash=\(flashSeconds)s noflash=\(noFlashSeconds)s (median of 3)")
    }

    func testLongAudio() async throws {
        let jfk = try BenchEnv.loadJFK()
        let long = BenchEnv.chain(jfk, count: 6, gapSeconds: 0)   // ~66 s
        let reference = BenchEnv.reference(times: 6)
        let engine = try await Self.sharedEngine()

        // Old default path (per-chunk z-norm, 20 s/2 s) vs. the new streamed
        // default (global z-norm, 30 s/5 s). The binary's own heuristic
        // (30 s/2 s) stays in the run as a watchdog: it loses words at chunk
        // boundaries today — if a future xcframework fixes it, its WER drops
        // to ~0 and transcribeLong could delegate to it again.
        let chunked = await engine.transcribeChunked(long, chunkSeconds: 20, overlapSeconds: 2)
        let streamed = await engine.transcribeLong(long)                              // 30/5
        let heuristic = await engine.transcribeLong(long, chunkSeconds: 0, overlapSeconds: -1)
        let chunkedWER = WordErrorRate.wer(reference: reference, hypothesis: chunked.text)
        let streamedWER = WordErrorRate.wer(reference: reference, hypothesis: streamed.text)
        let heuristicWER = WordErrorRate.wer(reference: reference, hypothesis: heuristic.text)

        BenchJSON.write([
            EngineBenchResult(name: "long-audio-chunked-20-2", wer: chunkedWER, transcript: chunked),
            EngineBenchResult(name: "long-audio-streamed-30-5-default", wer: streamedWER, transcript: streamed),
            EngineBenchResult(name: "long-audio-streamed-binary-heuristic", wer: heuristicWER, transcript: heuristic),
        ], name: "long-audio")

        XCTAssertLessThanOrEqual(streamedWER, chunkedWER + 0.02,
                                 "streamed default degraded vs chunked: \(streamed.text)")
        XCTAssertLessThanOrEqual(streamedWER, 0.30)
        // 66 s with 30 s chunks / 5 s overlap → 3 encoder passes; this also
        // catches a broken counter delta (streamed doesn't reset counters).
        XCTAssertEqual(streamed.encoderRuns, 3)
    }
}
#endif
