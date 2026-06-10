//
//  Diarizer.swift
//  Orchestrates native diarization: TitaNet embeddings + online clustering
//  (live, per committed utterance), pyannote segmentation + re-embedding
//  (final pass, word-level), and the optional persistent SpeakerDB for
//  named-speaker recognition. Cluster IDs are session-stable: live and final
//  pass share one clusterer.
//
#if os(iOS)
import Foundation
import ParakeetCore

public struct DiarizationOptions: Sendable {
    /// Cosine threshold for joining an existing speaker cluster (TitaNet: ~0.5).
    public var mergeThreshold: Float = 0.5
    /// Speaker-count budget: once this many clusters exist, every utterance is
    /// assigned to the nearest one — set it to the KNOWN number of speakers
    /// (e.g. 2 for an interview) to pin the session to exactly that many IDs.
    public var maxSpeakers: Int = 8
    /// Cosine threshold for recognising an enrolled (named) speaker.
    public var dbMatchThreshold: Float = 0.55
    /// Windows shorter than this are not embedded (unstable embeddings).
    public var minEmbeddingSeconds: Double = 0.6
    public var threads: Int = 2
    /// Run TitaNet/pyannote as ggml graphs on Metal. Default follows the
    /// engine heuristic: GPU on devices, CPU in the simulator.
    public var useGPU: Bool = ParakeetEngine.preferredUseGPU

    public init(mergeThreshold: Float = 0.5, maxSpeakers: Int = 8,
                dbMatchThreshold: Float = 0.55, minEmbeddingSeconds: Double = 0.6,
                threads: Int = 2, useGPU: Bool = ParakeetEngine.preferredUseGPU) {
        self.mergeThreshold = mergeThreshold
        self.maxSpeakers = max(1, maxSpeakers)
        self.dbMatchThreshold = dbMatchThreshold
        self.minEmbeddingSeconds = minEmbeddingSeconds
        self.threads = threads
        self.useGPU = useGPU
    }
}

public actor Diarizer {
    private let embedder: SpeakerEmbedder
    private let segmenter: PyannoteSegmenter?
    private let db: SpeakerDB?
    private let options: DiarizationOptions
    private var clusterer: SpeakerClusterer
    /// Cluster index → enrolled name (resolved once per cluster).
    private var names: [Int: String] = [:]

    public init(embedder: SpeakerEmbedder, segmenter: PyannoteSegmenter? = nil,
                db: SpeakerDB? = nil, options: DiarizationOptions = .init()) {
        self.embedder = embedder
        self.segmenter = segmenter
        self.db = db
        self.options = options
        self.clusterer = SpeakerClusterer(mergeThreshold: options.mergeThreshold,
                                          maxSpeakers: options.maxSpeakers)
    }

    /// Loads models off-main. `pyannoteModelPath` nil = no final pass;
    /// `speakerDBDirectory` nil = no named-speaker recognition.
    public static func make(titanetModelPath: String, pyannoteModelPath: String? = nil,
                            speakerDBDirectory: URL? = nil,
                            options: DiarizationOptions = .init()) async throws -> Diarizer {
        let embedder = try await SpeakerEmbedder.make(modelPath: titanetModelPath,
                                                      threads: options.threads,
                                                      useGPU: options.useGPU)
        var segmenter: PyannoteSegmenter? = nil
        if let path = pyannoteModelPath {
            segmenter = try await PyannoteSegmenter.make(modelPath: path, threads: options.threads,
                                                         useGPU: options.useGPU)
        }
        let db = try speakerDBDirectory.map { try SpeakerDB(directory: $0) }
        return Diarizer(embedder: embedder, segmenter: segmenter, db: db, options: options)
    }

    public var speakerCount: Int { clusterer.speakerCount }

    // MARK: - Live (utterance level)

    /// StreamingSession.SpeakerAttribution: committed window → (id, name).
    public func attribute(_ samples: [Float]) async -> (id: Int, name: String?)? {
        guard Double(samples.count) / 16_000 >= options.minEmbeddingSeconds else { return nil }
        guard let embedding = await embedder.embed(samples) else { return nil }
        let id = clusterer.assign(embedding)
        guard id >= 0 else { return nil }
        resolveName(for: id, embedding: embedding)
        return (id, names[id])
    }

    // MARK: - Final pass (word level)

    /// StreamingSession.SpeakerFinalize: pyannote turns over the full audio,
    /// one embedding per sufficiently long turn (clustered with the SAME
    /// session clusterer → IDs match the live ones), words labelled by
    /// best-overlap turn.
    public func finalize(audio: [Float], transcript: ParakeetTranscript) async -> ParakeetTranscript {
        guard let segmenter, !audio.isEmpty else { return transcript }
        guard let posteriors = await segmenter.posteriors(for: audio) else { return transcript }
        let localTurns = posteriors.speakerTurns(minTurnSeconds: options.minEmbeddingSeconds)
        guard !localTurns.isEmpty else { return transcript }

        // Embed + cluster each turn; remember the majority global ID per
        // pyannote-local speaker so short turns inherit it.
        var turnGlobal: [Int: Int] = [:]              // turn index → global id
        var localVotes: [Int: [Int: Int]] = [:]       // local id → global id → votes
        for (i, turn) in localTurns.enumerated() {
            let s = max(0, Int(turn.start * 16_000))
            let e = min(audio.count, Int(turn.end * 16_000))
            guard e > s, let embedding = await embedder.embed(Array(audio[s..<e])) else { continue }
            let id = clusterer.assign(embedding)
            guard id >= 0 else { continue }
            resolveName(for: id, embedding: embedding)
            turnGlobal[i] = id
            localVotes[turn.localSpeaker, default: [:]][id, default: 0] += 1
        }
        func inheritedGlobal(forLocal local: Int) -> Int? {
            localVotes[local]?.max(by: { $0.value < $1.value })?.key
        }

        var speakerTurns: [SpeakerTurn] = []
        for (i, turn) in localTurns.enumerated() {
            guard let id = turnGlobal[i] ?? inheritedGlobal(forLocal: turn.localSpeaker) else { continue }
            speakerTurns.append(SpeakerTurn(start: turn.start, end: turn.end,
                                            speaker: id, name: names[id]))
        }
        guard !speakerTurns.isEmpty else { return transcript }

        // Word → turn with the largest temporal overlap.
        let words = transcript.words.map { word -> ParakeetWord in
            var best: (overlap: Double, speaker: Int)? = nil
            for turn in speakerTurns {
                let overlap = min(word.end, turn.end) - max(word.start, turn.start)
                if overlap > 0, overlap > (best?.overlap ?? 0) {
                    best = (overlap, turn.speaker)
                }
            }
            return word.with(speaker: best?.speaker)
        }

        // Dominant speaker by labelled word time.
        var spoken: [Int: Double] = [:]
        for word in words where word.speaker != nil {
            spoken[word.speaker!, default: 0] += word.end - word.start
        }
        let dominant = spoken.max(by: { $0.value < $1.value })?.key

        return transcript.with(words: words, speaker: .some(dominant), speakerTurns: speakerTurns)
    }

    // MARK: - Enrollment / lifecycle

    /// Enrolls (or refines) a named speaker from a voice sample and persists
    /// it in the SpeakerDB. Throws when no DB is configured.
    public func enroll(name: String, samples: [Float]) async throws {
        guard let db else { throw ParakeetError.modelLoadFailed("speaker DB not configured") }
        guard let embedding = await embedder.embed(samples) else {
            throw ParakeetError.invalidEmbedding
        }
        try db.enroll(name: name, embedding: embedding)
        reresolveNames(db: db)
    }

    /// Names a SESSION cluster: persists its centroid (running mean over all
    /// of the speaker's utterances — typically more robust than a single
    /// sample) as a profile. Applies immediately to this instance (future
    /// `.speaker` events and the final pass); other instances pick the
    /// profile up when their SpeakerDB loads, i.e. from their next session.
    public func enrollCluster(id: Int, name: String) async throws {
        guard let db else { throw ParakeetError.modelLoadFailed("speaker DB not configured") }
        guard clusterer.clusters.indices.contains(id) else {
            throw ParakeetError.invalidEmbedding
        }
        try db.enroll(name: name, embedding: clusterer.clusters[id].centroid)
        reresolveNames(db: db)
        // Deterministic: after a running-mean refinement the threshold match
        // could fall just short — the named cluster keeps its name regardless.
        names[id] = name
    }

    /// Re-resolves the cluster→name cache against the DB (after enrollments).
    private func reresolveNames(db: SpeakerDB) {
        names.removeAll()
        for index in 0..<clusterer.speakerCount {
            if let match = db.match(clusterer.clusters[index].centroid,
                                    threshold: options.dbMatchThreshold) {
                names[index] = match.name
            }
        }
    }

    /// Forgets session speakers (clusters + name cache); the DB persists.
    public func reset() {
        clusterer.reset()
        names.removeAll()
    }

    private func resolveName(for id: Int, embedding: [Float]) {
        guard names[id] == nil, let db else { return }
        if let match = db.match(embedding, threshold: options.dbMatchThreshold) {
            names[id] = match.name
        }
    }
}

#endif
