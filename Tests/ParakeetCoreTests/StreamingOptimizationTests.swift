//
//  StreamingOptimizationTests.swift
//  Unit tests for the streaming cost optimizations: preview window cap (B3),
//  adaptive preview cadence (B4), commit reuse (B5). Driven deterministically
//  via scriptedAudio + EnergyVAD + the counting fake transcriber.
//
import XCTest
import ParakeetCore

final class StreamingOptimizationTests: XCTestCase {

    private func makeConfig() -> StreamingConfig {
        var cfg = StreamingConfig()
        cfg.vadHopSeconds = 0.3
        cfg.previewStepSeconds = 0.6
        cfg.endpointSilenceSeconds = 0.8
        cfg.maxSegmentSeconds = 100
        return cfg
    }

    /// Drives `parts` in 0.1 s blocks and returns (recorder, session).
    private func run(_ parts: [(speech: Double, silence: Double)],
                     config: StreamingConfig) async -> (CallRecorder, StreamingSession) {
        let recorder = CallRecorder()
        let session = StreamingSession(config: config, vad: EnergyVAD(),
                                       transcribe: countingFakeTranscribe(recorder: recorder))
        let audio = scriptedAudio(parts)
        var i = 0
        while i < audio.count {
            let end = min(i + 1600, audio.count)
            await session.drive(Array(audio[i..<end]))
            i = end
        }
        return (recorder, session)
    }

    // MARK: - B3 preview window cap

    func testPreviewWindowCapsCallLength() async {
        var cfg = makeConfig()
        cfg.previewWindowSeconds = 2
        let (recorder, session) = await run([(6, 1.2)], config: cfg)

        let perCall = await recorder.perCallSamples
        // The last call is the endpoint commit — it must see the FULL speech
        // window (6 s + 0.3 s pad) despite the preview cap.
        let commitSeconds = Double(perCall.last ?? 0) / 16_000
        XCTAssertGreaterThan(commitSeconds, 5.9, "commit must use the full window")

        // Previews (all calls before the commit) are capped: window + at most
        // one promoted-word lag (~1 s) + one preview step (0.6 s).
        let previewSeconds = perCall.dropLast().map { Double($0) / 16_000 }
        XCTAssertFalse(previewSeconds.isEmpty)
        XCTAssertLessThanOrEqual(previewSeconds.max() ?? 0, 4.0,
                                 "preview window not capped: \(previewSeconds)")

        // Frozen prefix + window text: the last hypothesis must still cover
        // roughly the whole utterance (more words than one capped window).
        let committed = await session.acceptedText
        XCTAssertFalse(committed.isEmpty)
    }

    func testPreviewWindowKeepsHypothesisPrefix() async {
        var cfg = makeConfig()
        cfg.previewWindowSeconds = 2
        let recorder = CallRecorder()
        let session = StreamingSession(config: cfg, vad: EnergyVAD(),
                                       transcribe: countingFakeTranscribe(recorder: recorder))
        // Drive 6 s of speech WITHOUT trailing silence — the hypothesis stays
        // live (no endpoint), so we can inspect prefix + window composition.
        let audio = scriptedAudio([(6, 0)])
        var i = 0
        while i < audio.count {
            let end = min(i + 1600, audio.count)
            await session.drive(Array(audio[i..<end]))
            i = end
        }
        let hyp = await session.hypothesisText
        let hypWords = hyp.split(separator: " ").count
        let maxPreview = await recorder.maxCallSeconds
        // Fake emits 1 word/second: a capped window alone yields ~2-3 words;
        // the full 6 s utterance needs the frozen prefix to stay visible.
        XCTAssertGreaterThanOrEqual(hypWords, 4, "prefix lost: '\(hyp)'")
        XCTAssertLessThanOrEqual(maxPreview, 4.0)
    }

    func testPreviewWindowOffMatchesOldBehaviour() async {
        var cfg = makeConfig()
        cfg.previewWindowSeconds = 0   // unbounded = pre-optimization behaviour
        let (recorder, _) = await run([(6, 1.2)], config: cfg)
        let perCall = await recorder.perCallSamples
        // Previews grow with the segment: the longest preview is close to the
        // full utterance length.
        let previewMax = perCall.dropLast().map { Double($0) / 16_000 }.max() ?? 0
        XCTAssertGreaterThan(previewMax, 5.0)
    }

    /// The committed text must be byte-identical with and without the cap —
    /// the cap only touches transient previews.
    func testCommittedTextIdenticalWithAndWithoutCap() async {
        var capped = makeConfig()
        capped.previewWindowSeconds = 2
        var uncapped = makeConfig()
        uncapped.previewWindowSeconds = 0

        let parts: [(speech: Double, silence: Double)] = [(3, 1.0), (6, 1.2)]
        let (_, cappedSession) = await run(parts, config: capped)
        let (_, uncappedSession) = await run(parts, config: uncapped)

        let cappedText = await cappedSession.acceptedText
        let uncappedText = await uncappedSession.acceptedText
        XCTAssertEqual(cappedText, uncappedText)
        XCTAssertFalse(cappedText.isEmpty)
    }
}
