//
//  StreamingSpeakerTests.swift
//  Live speaker attribution in StreamingSession: the injected
//  SpeakerAttribution closure runs per committed segment and surfaces as
//  `.speaker` events with matching segment indices.
//
import XCTest
import ParakeetCore

final class StreamingSpeakerTests: XCTestCase {

    private actor AttributionRecorder {
        private(set) var windows: [Int] = []
        func record(sampleCount: Int) -> Int {
            windows.append(sampleCount)
            return windows.count - 1   // speaker id = call order
        }
    }

    private func makeConfig() -> StreamingConfig {
        var cfg = StreamingConfig()
        cfg.vadHopSeconds = 0.3
        cfg.previewStepSeconds = 0.6
        cfg.endpointSilenceSeconds = 0.8
        cfg.maxSegmentSeconds = 100
        return cfg
    }

    /// Two utterances → two committed segments, each followed by a `.speaker`
    /// event whose segmentIndex matches the committed order.
    func testSpeakerEventsFollowCommits() async {
        let transcribeRecorder = CallRecorder()
        let attribution = AttributionRecorder()
        let session = StreamingSession(
            config: makeConfig(),
            vad: EnergyVAD(),
            transcribe: countingFakeTranscribe(recorder: transcribeRecorder),
            attributeSpeaker: { samples in
                let id = await attribution.record(sampleCount: samples.count)
                return (id: id, name: id == 0 ? "alice" : nil)
            })
        let events = await session.events()

        let audio = scriptedAudio([(2, 1.2), (2, 1.2)])
        var i = 0
        while i < audio.count {
            let end = min(i + 1600, audio.count)
            await session.drive(Array(audio[i..<end]))
            i = end
        }
        await session.commitRemaining()
        await session.finish()

        var committedIndices = 0
        var speakerEvents: [(segmentIndex: Int, id: Int, name: String?)] = []
        for await event in events {
            switch event {
            case .committed: committedIndices += 1
            case .speaker(let segmentIndex, let id, let name):
                speakerEvents.append((segmentIndex, id, name))
            default: break
            }
        }

        XCTAssertEqual(committedIndices, 2)
        XCTAssertEqual(speakerEvents.count, 2)
        XCTAssertEqual(speakerEvents.map(\.segmentIndex), [0, 1])
        XCTAssertEqual(speakerEvents.map(\.id), [0, 1])
        XCTAssertEqual(speakerEvents[0].name, "alice")
        XCTAssertNil(speakerEvents[1].name)

        // The attribution window covers the committed speech (≥ 2 s each).
        let windows = await attribution.windows
        XCTAssertTrue(windows.allSatisfy { $0 >= 2 * 16_000 }, "\(windows)")
    }

    /// Without the hook no `.speaker` events appear (default behaviour).
    func testNoSpeakerEventsWithoutAttribution() async {
        let recorder = CallRecorder()
        let session = StreamingSession(config: makeConfig(), vad: EnergyVAD(),
                                       transcribe: countingFakeTranscribe(recorder: recorder))
        let events = await session.events()
        let audio = scriptedAudio([(2, 1.2)])
        var i = 0
        while i < audio.count {
            let end = min(i + 1600, audio.count)
            await session.drive(Array(audio[i..<end]))
            i = end
        }
        await session.finish()

        for await event in events {
            if case .speaker = event { XCTFail("unexpected speaker event") }
        }
    }

    /// An undecided hook (nil) emits nothing — and empty commits never
    /// trigger attribution at all.
    func testUndecidedAttributionEmitsNoEvent() async {
        let recorder = CallRecorder()
        let session = StreamingSession(
            config: makeConfig(), vad: EnergyVAD(),
            transcribe: countingFakeTranscribe(recorder: recorder),
            attributeSpeaker: { _ in nil })
        let events = await session.events()
        let audio = scriptedAudio([(2, 1.2)])
        var i = 0
        while i < audio.count {
            let end = min(i + 1600, audio.count)
            await session.drive(Array(audio[i..<end]))
            i = end
        }
        await session.finish()

        var sawCommitted = false
        for await event in events {
            if case .committed = event { sawCommitted = true }
            if case .speaker = event { XCTFail("nil attribution must not emit") }
        }
        XCTAssertTrue(sawCommitted)
    }
}
