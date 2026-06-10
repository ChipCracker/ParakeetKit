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
