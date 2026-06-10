//
//  DiarizationBenchmarkTests.swift
//  Real-model diarization smoke + benchmark on two distinct synthetic
//  speakers (qwen3-tts CustomVoice "ryan" and "serena"): TitaNet embedding
//  consistency, two-voice separation, speaker-DB recognition, and the
//  pyannote word-level final pass. Models (44 MB + 6 MB) download once via
//  ModelDownloader.
//
#if os(iOS)
import XCTest
import ParakeetKit

final class DiarizationBenchmarkTests: XCTestCase {

    /// Same voice clusters together, the second voice gets its own ID, and an
    /// enrolled profile resolves the name.
    func testLiveAttributionSeparatesVoices() async throws {
        let titanet = try await BenchEnv.resolveDiarizationModelOrSkip(ParakeetModelCatalog.titanetLarge)
        let ryan = try BenchEnv.loadVoice("voice-ryan")
        let serena = try BenchEnv.loadVoice("voice-serena")
        let ryanFirstHalf = Array(ryan[0..<(ryan.count / 2)])
        let ryanSecondHalf = Array(ryan[(ryan.count / 2)...])

        let dbDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("diar-bench-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: dbDir) }

        let diarizer = try await Diarizer.make(titanetModelPath: titanet,
                                               speakerDBDirectory: dbDir)
        try await diarizer.enroll(name: "ryan", samples: ryanFirstHalf)

        let r1 = await diarizer.attribute(ryanFirstHalf)
        let r2 = await diarizer.attribute(ryanSecondHalf)
        let s = await diarizer.attribute(serena)

        XCTAssertNotNil(r1); XCTAssertNotNil(r2); XCTAssertNotNil(s)
        XCTAssertEqual(r1?.id, r2?.id, "same voice must share a speaker ID")
        XCTAssertNotEqual(r1?.id, s?.id, "second voice must get its own ID")
        XCTAssertEqual(r1?.name, "ryan", "enrolled profile must resolve")

        // Cluster naming (badge-tap flow): serena's centroid becomes a
        // profile, applies in-session AND persists for fresh instances.
        if let serenaID = s?.id {
            try await diarizer.enrollCluster(id: serenaID, name: "serena")
            let renamed = await diarizer.attribute(serena)
            XCTAssertEqual(renamed?.name, "serena", "cluster name must apply in-session")
            let fresh = try await Diarizer.make(titanetModelPath: titanet,
                                                speakerDBDirectory: dbDir)
            let recognised = await fresh.attribute(serena)
            XCTAssertEqual(recognised?.name, "serena", "profile must persist across instances")
        }

        BenchJSON.write(["ryanIDs": "\(r1?.id ?? -1)/\(r2?.id ?? -1)",
                         "serenaID": "\(s?.id ?? -1)",
                         "resolvedName": r1?.name ?? "nil"],
                        name: "diarization-live")
    }

    /// Final pass over a ryan → serena → ryan conversation: pyannote turns +
    /// word labels separate both speakers and re-identify the first one.
    func testFinalPassLabelsWords() async throws {
        let model = try await BenchEnv.resolveModelOrSkip()
        let titanet = try await BenchEnv.resolveDiarizationModelOrSkip(ParakeetModelCatalog.titanetLarge)
        let pyannote = try await BenchEnv.resolveDiarizationModelOrSkip(ParakeetModelCatalog.pyannoteSegmentation)

        let ryan = try BenchEnv.loadVoice("voice-ryan")
        let serena = try BenchEnv.loadVoice("voice-serena")
        let gap = [Float](repeating: 0, count: Int(0.8 * 16_000))
        let audio = ryan + gap + serena + gap + ryan

        let engine = try await ParakeetEngine.make(modelPath: model,
                                                   useGPU: ParakeetEngine.preferredUseGPU)
        let diarizer = try await Diarizer.make(titanetModelPath: titanet,
                                               pyannoteModelPath: pyannote)

        let t0 = Date()
        let transcript = await engine.transcribeLong(audio)
        let enriched = await diarizer.finalize(audio: audio, transcript: transcript)
        let seconds = Date().timeIntervalSince(t0)

        let labelled = enriched.words.filter { $0.speaker != nil }
        let distinctSpeakers = Set(labelled.compactMap(\.speaker))

        BenchJSON.write(["speakerTurns": "\(enriched.speakerTurns.count)",
                         "labelledWords": "\(labelled.count)/\(enriched.words.count)",
                         "distinctSpeakers": "\(distinctSpeakers.count)",
                         "finalPassSeconds": "\(seconds)"],
                        name: "diarization-final")

        XCTAssertGreaterThanOrEqual(enriched.speakerTurns.count, 3, "\(enriched.speakerTurns)")
        XCTAssertEqual(distinctSpeakers.count, 2,
                       "exactly two voices expected: \(enriched.speakerTurns)")
        XCTAssertGreaterThan(Double(labelled.count) / Double(max(1, enriched.words.count)), 0.7,
                             "most words should carry a speaker label")
        // Re-identification: first and last turn are the same physical voice.
        if let first = enriched.speakerTurns.first, let last = enriched.speakerTurns.last {
            XCTAssertEqual(first.speaker, last.speaker, "ryan must be re-identified after serena")
        }
        XCTAssertNotNil(enriched.speaker)
    }
}
#endif
