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

    func testSingleShotJFK() async throws {
        let model = try BenchEnv.requireModelOrSkip()
        let jfk = try BenchEnv.loadJFK()
        let engine = try await ParakeetEngine.make(modelPath: model,
                                                   useGPU: ParakeetEngine.preferredUseGPU)

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
        let model = try BenchEnv.requireModelOrSkip()
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
        let model = try BenchEnv.requireModelOrSkip()
        let jfk = try BenchEnv.loadJFK()
        let long = BenchEnv.chain(jfk, count: 6, gapSeconds: 0)   // ~66 s
        let reference = BenchEnv.reference(times: 6)
        let engine = try await ParakeetEngine.make(modelPath: model,
                                                   useGPU: ParakeetEngine.preferredUseGPU)

        let result = await engine.transcribeLong(long)
        let wer = WordErrorRate.wer(reference: reference, hypothesis: result.text)

        let bench = EngineBenchResult(name: "long-audio-transcribeLong", wer: wer, transcript: result)
        BenchJSON.write([bench], name: "long-audio")

        XCTAssertLessThanOrEqual(wer, 0.30, "long-form transcript degraded: \(result.text)")
    }
}
#endif
