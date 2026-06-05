import XCTest
import ParakeetCore

final class StreamingSessionTests: XCTestCase {

    // Mock transcriber: canned transcript with fixed run counts.
    private func mock(text: String, enc: Int = 1, dec: Int = 2) -> StreamingSession.Transcribe {
        { samples in
            ParakeetTranscript(text: text, words: [], encoderRuns: enc, decoderSteps: dec,
                               audioSeconds: Double(samples.count) / 16_000, processingSeconds: 0.01)
        }
    }

    // Mock VAD: maps sample count → last-speech-end (seconds).
    private struct MockVAD: VADGating {
        let end: @Sendable (Int) -> Double
        func detect(_ samples: [Float], threshold: Float, minSpeech: Float, minSilence: Float) async -> [SpeechSegment] {
            [SpeechSegment(start: 0, end: end(samples.count))]
        }
    }

    private func samples(_ seconds: Double) -> [Float] {
        Array(repeating: 0.1, count: Int(seconds * 16_000))
    }

    func testHypothesisWhileSpeakingNotCommitted() async {
        var cfg = StreamingConfig()
        cfg.maxSegmentSeconds = 100; cfg.previewStepSeconds = 0.1; cfg.vadHopSeconds = 0.1
        // Still speaking: lastSpeechEnd == bufDuration → trailingSilence 0 → no endpoint.
        let vad = MockVAD(end: { Double($0) / 16_000 })
        let session = StreamingSession(config: cfg, vad: vad, transcribe: mock(text: "hyp"))

        await session.drive(samples(0.2))

        let hyp = await session.hypothesisText
        let acc = await session.acceptedText
        let stats = await session.stats
        XCTAssertEqual(hyp, "hyp")          // preview produced
        XCTAssertEqual(acc, "")             // nothing committed
        XCTAssertEqual(stats.encoderRuns, 0)  // hypotheses don't accumulate runs
    }

    func testCommitOnSilenceAccumulatesRuns() async {
        var cfg = StreamingConfig()
        cfg.endpointSilenceSeconds = 0.5; cfg.maxSegmentSeconds = 100; cfg.vadHopSeconds = 0.1
        // trailingSilence = bufDuration - lastSpeechEnd = 0.8 >= 0.5 → commit.
        let vad = MockVAD(end: { max(0, Double($0) / 16_000 - 0.8) })
        let session = StreamingSession(config: cfg, vad: vad, transcribe: mock(text: "done", enc: 3, dec: 5))

        await session.drive(samples(1.0))

        let acc = await session.acceptedText
        let hyp = await session.hypothesisText
        XCTAssertEqual(acc, "done")
        XCTAssertEqual(hyp, "")
        let stats = await session.stats
        XCTAssertEqual(stats.encoderRuns, 3)
        XCTAssertEqual(stats.decoderSteps, 5)
    }

    func testCommitOnMaxSegmentWithVADOff() async {
        var cfg = StreamingConfig()
        cfg.maxSegmentSeconds = 0.5; cfg.vadHopSeconds = 0.1; cfg.endpointSilenceSeconds = 100
        let session = StreamingSession(config: cfg, vad: NoOpVADGate(), transcribe: mock(text: "max"))

        await session.drive(samples(0.6))   // bufDuration 0.6 >= maxSegment 0.5 → commit

        let acc = await session.acceptedText
        XCTAssertEqual(acc, "max")
    }

    func testEventsStreamEmitsCommittedAndFinalized() async {
        var cfg = StreamingConfig()
        cfg.maxSegmentSeconds = 0.5; cfg.vadHopSeconds = 0.1; cfg.endpointSilenceSeconds = 100
        let session = StreamingSession(config: cfg, vad: NoOpVADGate(), transcribe: mock(text: "evt"))

        let stream = await session.events()
        await session.drive(samples(0.6))
        await session.finish()

        var committedFull: String?
        var sawFinalized = false
        for await event in stream {
            switch event {
            case .committed(_, let full): committedFull = full
            case .finalized: sawFinalized = true
            default: break
            }
        }
        XCTAssertEqual(committedFull, "evt")
        XCTAssertTrue(sawFinalized)
    }
}
