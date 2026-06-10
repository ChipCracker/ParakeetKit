//
//  StreamingPipelineBenchmarkTests.swift
//  Deterministic cost profile of the streaming pipeline (no binary): counts
//  how many transcribe calls / audio-seconds the session spends per scenario.
//  Run via `swift test --filter StreamingPipelineBenchmarkTests`; benchmark.sh
//  collects the JSON written to $PARAKEET_BENCH_OUT.
//
import XCTest
import ParakeetCore

final class StreamingPipelineBenchmarkTests: XCTestCase {

    /// Three utterances: short (4 s), medium (8 s), long (16 s — exercises the
    /// maxSegment commit path). Pauses are long enough for endpoint commits.
    private let scenarioParts: [(speech: Double, silence: Double)] = [(4, 1.0), (8, 1.0), (16, 1.2)]

    private func runScenario(name: String, parts: [(speech: Double, silence: Double)],
                             config: StreamingConfig) async -> PipelineBenchResult {
        let recorder = CallRecorder()
        let session = StreamingSession(config: config, vad: EnergyVAD(),
                                       transcribe: countingFakeTranscribe(recorder: recorder))
        let audio = scriptedAudio(parts)
        var i = 0
        let chunk = 1600 // 0.1 s blocks, like the mic callback
        while i < audio.count {
            let end = min(i + chunk, audio.count)
            await session.drive(Array(audio[i..<end]))
            i = end
        }
        await session.commitRemaining()

        // stats accumulates commit runs only (fake: encoderRuns == 1 per call),
        // the recorder sees every call — previews are the difference.
        let stats = await session.stats
        let calls = await recorder.calls
        let totalSeconds = await recorder.audioSeconds
        let commitCalls = stats.encoderRuns
        return PipelineBenchResult(
            scenario: name,
            transcribeCalls: calls,
            previewCalls: calls - commitCalls,
            commitCalls: commitCalls,
            previewAudioSeconds: totalSeconds - stats.audioSeconds,
            commitAudioSeconds: stats.audioSeconds,
            ingestedAudioSeconds: Double(audio.count) / 16_000,
            committedText: await session.acceptedText)
    }

    func testPipelineCostDefaults() async {
        // Current defaults vs. the pre-optimization configuration (no preview
        // cap, fixed cadence) — the committed text must be identical, only
        // the preview cost may differ.
        var legacy = StreamingConfig()
        legacy.previewWindowSeconds = 0
        legacy.previewSlowAfterSeconds = .infinity
        legacy.reuseLastPreviewOnCommit = false

        let result = await runScenario(name: "defaults", parts: scenarioParts,
                                       config: StreamingConfig())
        let legacyResult = await runScenario(name: "legacy-unbounded", parts: scenarioParts,
                                             config: legacy)
        print(PipelineBenchResult.markdownHeader)
        print(result.markdownRow)
        print(legacyResult.markdownRow)
        BenchOutput.write([result, legacyResult], name: "pipeline-benchmark")

        XCTAssertEqual(result.committedText, legacyResult.committedText,
                       "optimizations changed the committed text")
        XCTAssertFalse(result.committedText.isEmpty)
        XCTAssertGreaterThanOrEqual(result.commitCalls, 3)   // one per utterance minimum
        XCTAssertGreaterThan(result.previewCalls, 0)
        XCTAssertLessThanOrEqual(result.previewAudioSeconds, legacyResult.previewAudioSeconds)
    }
}
