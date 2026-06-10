//
//  E2EStreamingBenchmarkTests.swift
//  Full live-pipeline benchmark: StreamingSession + real engine + bundled
//  FireRedVAD over jfk ×3 (1.2 s gaps → deterministic endpoints). A wrapper
//  closure records EVERY transcribe call (previews + commits) — the session's
//  own stats only cover commits.
//
#if os(iOS)
import XCTest
import ParakeetKit

final class E2EStreamingBenchmarkTests: XCTestCase {

    func testE2EStreaming() async throws {
        let model = try await BenchEnv.resolveModelOrSkip()
        let jfk = try BenchEnv.loadJFK()
        let audio = BenchEnv.chain(jfk, count: 3, gapSeconds: 1.2)   // ~35 s
        let reference = BenchEnv.reference(times: 3)

        let engine = try await ParakeetEngine.make(modelPath: model,
                                                   useGPU: ParakeetEngine.preferredUseGPU)
        let vad = try await FireRedVAD.bundled()
        let recorder = BenchCallRecorder()

        let session = StreamingSession(
            config: StreamingConfig(),
            vad: vad,
            transcribe: { samples in
                let r = await engine.transcribe(samples)
                await recorder.record(sampleCount: samples.count, processing: r.processingSeconds)
                return r
            },
            transcribeLong: { samples in
                let r = await engine.transcribeLong(samples)
                await recorder.record(sampleCount: samples.count, processing: r.processingSeconds)
                return r
            })

        let events = await session.events()   // exactly once, before driving

        var i = 0
        while i < audio.count {                // 0.1 s blocks, like the mic tap
            let end = min(i + 1600, audio.count)
            await session.drive(Array(audio[i..<end]))
            i = end
        }
        await session.commitRemaining()
        let committed = await session.acceptedText
        let stats = await session.stats
        await session.finish()                 // long pass + closes the stream

        var finalized = ""
        for await event in events {
            if case .finalized(let text) = event { finalized = text }
        }

        let committedWER = WordErrorRate.wer(reference: reference, hypothesis: committed)
        let finalizedWER = WordErrorRate.wer(reference: reference, hypothesis: finalized)

        var bench = EngineBenchResult(
            name: "e2e-streaming-jfk-x3",
            wer: committedWER,
            transcript: ParakeetTranscript(text: committed, words: [],
                                           encoderRuns: stats.encoderRuns,
                                           decoderSteps: stats.decoderSteps,
                                           audioSeconds: stats.audioSeconds,
                                           processingSeconds: stats.processingSeconds))
        bench.transcribeCalls = await recorder.calls
        bench.transcribedSecondsTotal = await recorder.audioSeconds
        bench.totalProcessingSeconds = await recorder.processingSeconds
        bench.committedWER = committedWER
        bench.finalizedWER = finalizedWER
        BenchJSON.write(bench, name: "e2e-streaming")

        XCTAssertLessThanOrEqual(committedWER, 0.15, "committed degraded: \(committed)")
        XCTAssertLessThanOrEqual(finalizedWER, 0.15, "finalized degraded: \(finalized)")
        XCTAssertFalse(committed.isEmpty)
    }
}
#endif
