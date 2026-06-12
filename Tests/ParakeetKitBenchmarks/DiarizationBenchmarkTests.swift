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

    /// CPU↔GPU parity of the graph ports (devices only — the simulator runs
    /// CPU like the engine): TitaNet embeddings cosine ≥ 0.999, pyannote
    /// turn sequences identical (±1 frame), GPU times recorded.
    /// PARAKEET_BENCH_FORCE_GPU=1 overrides the skip — intended for unusual
    /// device setups. In the SIMULATOR this crashes by design of the platform:
    /// MTLSimDriver cannot create ggml-metal's shared-memory buffers
    /// (xpc_shmem_create → xpc_api_misuse in newBufferWithLength), which is
    /// exactly why preferredUseGPU is false there.
    func testGPUParity() async throws {
        guard ParakeetEngine.preferredUseGPU
                || ProcessInfo.processInfo.environment["PARAKEET_BENCH_FORCE_GPU"] != nil else {
            throw XCTSkip("GPU parity runs on physical devices only (or PARAKEET_BENCH_FORCE_GPU=1)")
        }
        let titanet = try await BenchEnv.resolveDiarizationModelOrSkip(ParakeetModelCatalog.titanetLarge)
        let pyannote = try await BenchEnv.resolveDiarizationModelOrSkip(ParakeetModelCatalog.pyannoteSegmentation)
        let ryan = try BenchEnv.loadVoice("voice-ryan")
        let serena = try BenchEnv.loadVoice("voice-serena")
        let gap = [Float](repeating: 0, count: Int(0.8 * 16_000))
        let audio = ryan + gap + serena + gap + ryan

        // TitaNet: same voice, both paths → near-identical embedding.
        let embCPU = try await SpeakerEmbedder.make(modelPath: titanet, useGPU: false)
        let embGPU = try await SpeakerEmbedder.make(modelPath: titanet, useGPU: true)
        guard let eCPU = await embCPU.embed(ryan), var eGPU = await embGPU.embed(ryan) else {
            return XCTFail("embedding failed")
        }
        var t0 = Date()
        for _ in 0..<3 { eGPU = await embGPU.embed(ryan) ?? eGPU }
        let titanetGPUSeconds = Date().timeIntervalSince(t0) / 3
        let cos = zip(eCPU, eGPU).reduce(Float(0)) { $0 + $1.0 * $1.1 } // both L2-normalized
        XCTAssertGreaterThanOrEqual(cos, 0.999, "titanet cpu/gpu drifted: \(cos)")

        // Pyannote: identical turn sequence (local speakers + ±1 frame times).
        let segCPU = try await PyannoteSegmenter.make(modelPath: pyannote, useGPU: false)
        let segGPU = try await PyannoteSegmenter.make(modelPath: pyannote, useGPU: true)
        guard let pCPU = await segCPU.posteriors(for: audio) else { return XCTFail("cpu seg failed") }
        t0 = Date()
        guard let pGPU = await segGPU.posteriors(for: audio) else { return XCTFail("gpu seg failed") }
        let pyannoteGPUSeconds = Date().timeIntervalSince(t0)
        let turnsCPU = pCPU.speakerTurns()
        let turnsGPU = pGPU.speakerTurns()
        XCTAssertEqual(turnsCPU.count, turnsGPU.count, "\(turnsCPU) vs \(turnsGPU)")
        let frame = PyannotePosteriors.frameDuration
        for (a, b) in zip(turnsCPU, turnsGPU) {
            XCTAssertEqual(a.localSpeaker, b.localSpeaker)
            XCTAssertEqual(a.start, b.start, accuracy: frame * 1.5)
            XCTAssertEqual(a.end, b.end, accuracy: frame * 1.5)
        }

        BenchJSON.write(["titanetCosine": "\(cos)",
                         "titanetGPUSeconds": "\(titanetGPUSeconds)",
                         "pyannoteGPUSeconds": "\(pyannoteGPUSeconds)",
                         "turns": "\(turnsGPU.count)"],
                        name: "diarization-gpu-parity")
    }

    /// Three voices with recurrence (ryan serena aiden ryan serena): word
    /// labels are scored against the known concatenation ground truth. The
    /// majority mapping cluster→voice must be a bijection over three
    /// clusters (recurrence re-identified) and the word-label accuracy must
    /// clear 0.85.
    func testFinalPassThreeSpeakers() async throws {
        let model = try await BenchEnv.resolveModelOrSkip()
        let titanet = try await BenchEnv.resolveDiarizationModelOrSkip(ParakeetModelCatalog.titanetLarge)
        let pyannote = try await BenchEnv.resolveDiarizationModelOrSkip(ParakeetModelCatalog.pyannoteSegmentation)

        let voices = [try BenchEnv.loadVoice("voice-ryan"),
                      try BenchEnv.loadVoice("voice-serena"),
                      try BenchEnv.loadVoice("voice-aiden")]
        let order = [0, 1, 2, 0, 1]
        let gap = [Float](repeating: 0, count: Int(0.8 * 16_000))

        var audio: [Float] = []
        var truth: [(start: Double, end: Double, voice: Int)] = []
        for (i, voice) in order.enumerated() {
            if i > 0 { audio += gap }
            let start = Double(audio.count) / 16_000
            audio += voices[voice]
            truth.append((start, Double(audio.count) / 16_000, voice))
        }

        let engine = try await ParakeetEngine.make(modelPath: model,
                                                   useGPU: ParakeetEngine.preferredUseGPU)
        let diarizer = try await Diarizer.make(titanetModelPath: titanet,
                                               pyannoteModelPath: pyannote)

        let t0 = Date()
        let transcript = await engine.transcribeLong(audio)
        let enriched = await diarizer.finalize(audio: audio, transcript: transcript)
        let seconds = Date().timeIntervalSince(t0)

        func truthVoice(at time: Double) -> Int? {
            truth.first(where: { time >= $0.start && time < $0.end })?.voice
        }
        var votes: [Int: [Int: Int]] = [:]                 // voice → cluster → count
        var scored: [(voice: Int, cluster: Int)] = []
        var inSpeech = 0
        for word in enriched.words {
            let mid = (word.start + word.end) / 2
            guard let voice = truthVoice(at: mid) else { continue }
            inSpeech += 1
            guard let cluster = word.speaker else { continue }
            votes[voice, default: [:]][cluster, default: 0] += 1
            scored.append((voice, cluster))
        }
        let mapping = votes.compactMapValues { $0.max(by: { $0.value < $1.value })?.key }
        let correct = scored.filter { mapping[$0.voice] == $0.cluster }.count
        let accuracy = scored.isEmpty ? 0 : Double(correct) / Double(scored.count)
        let coverage = inSpeech == 0 ? 0 : Double(scored.count) / Double(inSpeech)

        BenchJSON.write(["accuracy": String(format: "%.3f", accuracy),
                         "coverage": String(format: "%.3f", coverage),
                         "mappedClusters": "\(Set(mapping.values).count)",
                         "turns": "\(enriched.speakerTurns.count)",
                         "finalPassSeconds": String(format: "%.2f", seconds)],
                        name: "diarization-final-3spk")

        XCTAssertEqual(Set(mapping.values).count, 3,
                       "three voices must map to three distinct clusters: \(votes)")
        XCTAssertGreaterThanOrEqual(accuracy, 0.85, "word-label accuracy \(accuracy), votes \(votes)")
        XCTAssertGreaterThanOrEqual(coverage, 0.7, "labelled share of in-speech words: \(coverage)")
    }

    /// REAL concurrent speech: dylan and sohee are mixed so sohee starts
    /// 2 s before dylan ends. The v2 final pass must (a) express the zone as
    /// time-overlapping turns of two speakers — the old argmax pass could
    /// never produce that — and (b) keep the voices cleanly apart on their
    /// solo stretches despite the contaminated middle.
    func testFinalPassOverlappingSpeech() async throws {
        let model = try await BenchEnv.resolveModelOrSkip()
        let titanet = try await BenchEnv.resolveDiarizationModelOrSkip(ParakeetModelCatalog.titanetLarge)
        let pyannote = try await BenchEnv.resolveDiarizationModelOrSkip(ParakeetModelCatalog.pyannoteSegmentation)

        // TTS clips carry leading/trailing silence — trim it so the
        // constructed overlap is REAL concurrent speech.
        func trimmed(_ samples: [Float], threshold: Float = 0.01) -> [Float] {
            guard let lead = samples.firstIndex(where: { abs($0) > threshold }),
                  let trail = samples.lastIndex(where: { abs($0) > threshold }),
                  trail > lead else { return samples }
            return Array(samples[lead...trail])
        }
        let dylan = trimmed(try BenchEnv.loadVoice("voice-dylan"))
        let sohee = trimmed(try BenchEnv.loadVoice("voice-sohee"))
        let overlapSeconds = 3.0
        let dylanEnd = Double(dylan.count) / 16_000
        let soheeStart = dylanEnd - overlapSeconds
        let offset = Int(soheeStart * 16_000)

        var audio = [Float](repeating: 0, count: max(dylan.count, offset + sohee.count))
        for (i, sample) in dylan.enumerated() { audio[i] += sample }
        for (i, sample) in sohee.enumerated() { audio[offset + i] += sample }
        for i in 0..<audio.count { audio[i] = max(-1, min(1, audio[i])) }
        let soheeEnd = Double(offset + sohee.count) / 16_000

        // Solo ground truth with a safety margin around the overlap zone.
        let margin = 0.3
        let dylanSolo = (start: 0.0, end: soheeStart - margin)
        let soheeSolo = (start: dylanEnd + margin, end: soheeEnd)

        let engine = try await ParakeetEngine.make(modelPath: model,
                                                   useGPU: ParakeetEngine.preferredUseGPU)
        let diarizer = try await Diarizer.make(titanetModelPath: titanet,
                                               pyannoteModelPath: pyannote)
        let transcript = await engine.transcribeLong(audio)
        let enriched = await diarizer.finalize(audio: audio, transcript: transcript)

        // (a) Overlapping turns of DIFFERENT speakers.
        let turns = enriched.speakerTurns
        var maxTurnOverlap = 0.0
        for i in 0..<turns.count {
            for k in (i + 1)..<turns.count where turns[i].speaker != turns[k].speaker {
                let overlap = min(turns[i].end, turns[k].end) - max(turns[i].start, turns[k].start)
                maxTurnOverlap = max(maxTurnOverlap, overlap)
            }
        }

        // (b) Word mapping on the solo stretches.
        var votes: [Int: [Int: Int]] = [:]                 // gt voice → cluster → n
        var scored: [(gt: Int, cluster: Int)] = []
        for word in enriched.words {
            let mid = (word.start + word.end) / 2
            let gt: Int
            if mid >= dylanSolo.start && mid < dylanSolo.end { gt = 0 }
            else if mid >= soheeSolo.start && mid < soheeSolo.end { gt = 1 }
            else { continue }
            guard let cluster = word.speaker else { continue }
            votes[gt, default: [:]][cluster, default: 0] += 1
            scored.append((gt, cluster))
        }
        let mapping = votes.compactMapValues { $0.max(by: { $0.value < $1.value })?.key }
        let correct = scored.filter { mapping[$0.gt] == $0.cluster }.count
        let soloAccuracy = scored.isEmpty ? 0 : Double(correct) / Double(scored.count)

        BenchJSON.write(["maxTurnOverlapSeconds": String(format: "%.2f", maxTurnOverlap),
                         "soloAccuracy": String(format: "%.3f", soloAccuracy),
                         "mappedClusters": "\(Set(mapping.values).count)",
                         "turns": "\(turns.count)"],
                        name: "diarization-final-overlap")

        XCTAssertGreaterThanOrEqual(maxTurnOverlap, 0.5,
                                    "concurrent speech must yield overlapping turns: \(turns)")
        XCTAssertEqual(Set(mapping.values).count, 2,
                       "two voices on the solo stretches: \(votes)")
        XCTAssertGreaterThanOrEqual(soloAccuracy, 0.9,
                                    "solo accuracy \(soloAccuracy), votes \(votes)")
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
