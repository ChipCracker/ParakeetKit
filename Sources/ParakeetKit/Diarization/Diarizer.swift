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

    // MARK: - Final pass (word level), v2

    /// Final-pass tuning. Constants by design — the bench guards them.
    private enum Tuning {
        static let segmentationWindowSeconds = 10.0  // pyannote runs windowed
        static let minTurnSeconds = 0.3        // keep short turns (they inherit)
        static let subWindowMinTurn = 6.0      // longer turns embed in windows
        static let subWindowLength = 4.0
        static let subWindowHop = 2.0
        static let maxWindowsPerTurn = 6
        static let embeddingBudget = 96        // hard cap on embed() calls
        static let nearestWordSeconds = 0.5    // fallback for unlabelled words
        static let sameSpeakerMergeGap = 0.3   // heal seams at window borders
    }

    /// StreamingSession.SpeakerFinalize, v2: overlap-aware pyannote turns
    /// (one per concurrently active speaker), purity-masked windowed TitaNet
    /// embeddings, OFFLINE agglomerative clustering over all windows (the
    /// online clusterer stays live-only), conservative turn splitting,
    /// ID-mapping onto the session clusters (live IDs stay valid), activity
    /// tie-breaking for overlapped words plus a nearest-turn fallback, and
    /// centroid-based conflict-free name resolution.
    public func finalize(audio: [Float], transcript: ParakeetTranscript) async -> ParakeetTranscript {
        guard let segmenter, !audio.isEmpty else { return transcript }

        // pyannote runs in 10 s windows: its activations grow with T, so a
        // single full-length pass over long recordings (a 17-minute take is
        // ~0.5–1 GB of graph buffers) gets the app jetsam-killed — and 10 s
        // is the model's training domain anyway. Local speaker slots are
        // only meaningful per window; global identity comes from the
        // embeddings, and the same-speaker merge heals the window seams.
        let windowSamples = Int(Tuning.segmentationWindowSeconds * 16_000)
        var segmentWindows: [PyannotePosteriors] = []
        var cursor = 0
        while cursor < audio.count {
            let end = min(cursor + windowSamples, audio.count)
            // A sub-second tail yields no stable turns — skip it.
            if end - cursor < 16_000, !segmentWindows.isEmpty { break }
            if let posterior = await segmenter.posteriors(for: Array(audio[cursor..<end]),
                                                          startTime: Double(cursor) / 16_000) {
                segmentWindows.append(posterior)
            }
            cursor = end
        }
        guard !segmentWindows.isEmpty else { return transcript }

        var localTurns: [(start: Double, end: Double, localSpeaker: Int, windowIndex: Int)] = []
        for (windowIndex, posterior) in segmentWindows.enumerated() {
            for turn in posterior.perSpeakerTurns(minTurnSeconds: Tuning.minTurnSeconds) {
                localTurns.append((turn.start, turn.end, turn.localSpeaker, windowIndex))
            }
        }
        localTurns.sort { $0.start < $1.start }
        guard !localTurns.isEmpty else { return transcript }

        // --- Embedding windows: purity-masked, budgeted ---------------------
        struct Window {
            let turnIndex: Int
            let start: Double
            let end: Double
            var mid: Double { (start + end) / 2 }
            var duration: Double { end - start }
        }
        var windows: [Window] = []
        for (i, turn) in localTurns.enumerated() {
            let duration = turn.end - turn.start
            if duration < Tuning.subWindowMinTurn {
                let range = segmentWindows[turn.windowIndex]
                    .pureRange(forLocal: turn.localSpeaker,
                               from: turn.start, to: turn.end,
                               minSeconds: options.minEmbeddingSeconds)
                    ?? (start: turn.start, end: turn.end)
                windows.append(Window(turnIndex: i, start: range.start, end: range.end))
            } else {
                var hop = Tuning.subWindowHop
                var count = Int((duration - Tuning.subWindowLength) / hop) + 1
                if count > Tuning.maxWindowsPerTurn {
                    count = Tuning.maxWindowsPerTurn
                    hop = (duration - Tuning.subWindowLength) / Double(count - 1)
                }
                for k in 0..<count {
                    let start = turn.start + Double(k) * hop
                    windows.append(Window(turnIndex: i, start: start,
                                          end: min(start + Tuning.subWindowLength, turn.end)))
                }
            }
        }
        // Budget: shave windows from the most-windowed turns first, so every
        // turn keeps at least one.
        while windows.count > Tuning.embeddingBudget {
            var perTurn: [Int: [Int]] = [:]
            for (k, window) in windows.enumerated() {
                perTurn[window.turnIndex, default: []].append(k)
            }
            guard let richest = perTurn.max(by: { $0.value.count < $1.value.count }),
                  richest.value.count > 1 else { break }
            windows.remove(at: richest.value[richest.value.count / 2])
        }

        // --- Embed (≥ minEmbeddingSeconds only; short turns inherit later) --
        var embeddings: [[Float]] = []
        var meta: [(turnIndex: Int, mid: Double, duration: Double)] = []
        let minSamples = Int(options.minEmbeddingSeconds * 16_000)
        for window in windows {
            let s = max(0, Int(window.start * 16_000))
            let e = min(audio.count, Int(window.end * 16_000))
            guard e - s >= minSamples,
                  let embedding = await embedder.embed(Array(audio[s..<e])) else { continue }
            embeddings.append(embedding)
            meta.append((window.turnIndex, window.mid, window.duration))
        }
        guard !embeddings.isEmpty else { return transcript }

        // --- Offline clustering over ALL windows ----------------------------
        let labels = SpeakerClusterer.agglomerate(embeddings,
                                                  stopThreshold: options.mergeThreshold,
                                                  maxClusters: options.maxSpeakers)

        // Per-turn label: duration-weighted majority; conservative split on a
        // clean prefix/suffix label change (≥ 2 windows on each side).
        struct ResolvedTurn {
            var start: Double
            var end: Double
            let localSpeaker: Int
            let windowIndex: Int
            var label: Int?
        }
        var resolved: [ResolvedTurn] = localTurns.map {
            ResolvedTurn(start: $0.start, end: $0.end, localSpeaker: $0.localSpeaker,
                         windowIndex: $0.windowIndex, label: nil)
        }
        var splits: [ResolvedTurn] = []
        for index in resolved.indices {
            let mine = meta.enumerated()
                .filter { $0.element.turnIndex == index && labels[$0.offset] >= 0 }
                .sorted { $0.element.mid < $1.element.mid }
            guard !mine.isEmpty else { continue }
            let sequence = mine.map { labels[$0.offset] }
            if let cut = cleanLabelChange(sequence) {
                let splitTime = (mine[cut - 1].element.mid + mine[cut].element.mid) / 2
                var tail = resolved[index]
                tail.start = splitTime
                tail.label = sequence.last
                resolved[index].end = splitTime
                resolved[index].label = sequence.first
                splits.append(tail)
            } else {
                var weight: [Int: Double] = [:]
                for entry in mine {
                    weight[labels[entry.offset], default: 0] += entry.element.duration
                }
                resolved[index].label = weight.max(by: { $0.value < $1.value })?.key
            }
        }
        resolved += splits
        // Short/unembedded turns inherit the duration-weighted majority label
        // of their LOCAL pyannote speaker.
        var localWeight: [Int: [Int: Double]] = [:]
        for turn in resolved where turn.label != nil {
            localWeight[turn.localSpeaker, default: [:]][turn.label!, default: 0] += turn.end - turn.start
        }
        for index in resolved.indices where resolved[index].label == nil {
            resolved[index].label = localWeight[resolved[index].localSpeaker]?
                .max(by: { $0.value < $1.value })?.key
        }
        resolved = resolved.filter { $0.label != nil }.sorted { $0.start < $1.start }
        guard !resolved.isEmpty else { return transcript }

        // --- Map AHC clusters onto session clusters (live IDs stay valid) ---
        var clusterSum: [Int: [Float]] = [:]
        var clusterDuration: [Int: Double] = [:]
        var clusterFirst: [Int: Double] = [:]
        for (k, label) in labels.enumerated() where label >= 0 {
            if clusterSum[label] == nil {
                clusterSum[label] = embeddings[k]
            } else {
                for j in 0..<min(clusterSum[label]!.count, embeddings[k].count) {
                    clusterSum[label]![j] += embeddings[k][j]
                }
            }
            clusterDuration[label, default: 0] += meta[k].duration
            clusterFirst[label] = min(clusterFirst[label] ?? .infinity, meta[k].mid)
        }
        let sessionWasEmpty = clusterer.speakerCount == 0
        let mappingOrder = clusterSum.keys.sorted {
            sessionWasEmpty
                ? clusterFirst[$0]! < clusterFirst[$1]!                  // stable: first heard = id 0
                : clusterDuration[$0]! > clusterDuration[$1]!            // big clusters claim live ids first
        }
        var globalForLabel: [Int: Int] = [:]
        var takenLive = Set<Int>()
        for label in mappingOrder {
            let centroid = clusterSum[label]!
            var assigned: Int? = nil
            if !sessionWasEmpty {
                var bestSim = options.mergeThreshold
                for liveIndex in 0..<clusterer.speakerCount where !takenLive.contains(liveIndex) {
                    if let sim = clusterer.similarity(of: centroid, toSpeaker: liveIndex), sim >= bestSim {
                        bestSim = sim
                        assigned = liveIndex
                    }
                }
            }
            if let live = assigned {
                takenLive.insert(live)
                globalForLabel[label] = live
            } else {
                let id = clusterer.assign(centroid)
                if id >= 0 {
                    takenLive.insert(id)
                    globalForLabel[label] = id
                }
            }
        }

        // --- Names: centroid vs. DB, conflict-free; manual names win --------
        if let db {
            var claims: [(id: Int, name: String, score: Float)] = []
            for (label, id) in globalForLabel where names[id] == nil {
                if let match = db.match(clusterSum[label]!, threshold: options.dbMatchThreshold) {
                    claims.append((id, match.name, match.score))
                }
            }
            for claim in claims.sorted(by: { $0.score > $1.score })
            where names[claim.id] == nil && !names.values.contains(claim.name) {
                names[claim.id] = claim.name
            }
        }

        // --- Speaker turns (may overlap in time) ----------------------------
        struct GlobalTurn {
            let start: Double
            let end: Double
            let localSpeaker: Int
            let windowIndex: Int
            let speaker: Int
        }
        let rawGlobalTurns: [GlobalTurn] = resolved.compactMap { turn in
            guard let label = turn.label, let id = globalForLabel[label] else { return nil }
            return GlobalTurn(start: turn.start, end: turn.end,
                              localSpeaker: turn.localSpeaker,
                              windowIndex: turn.windowIndex, speaker: id)
        }
        // Same-speaker turns that touch or overlap merge into one — pyannote
        // sometimes tracks the SAME voice on two local slots (identical
        // activity from a shared powerset class), and the 10 s segmentation
        // windows cut continuing turns at their borders (sub-minTurn
        // snippets may vanish there). The gap tolerance heals those seams;
        // cross-speaker overlaps stay. Merged turns keep the first window's
        // index (activity queries clamp into it).
        var globalTurns: [GlobalTurn] = []
        for turn in rawGlobalTurns.sorted(by: { $0.start < $1.start }) {
            if let last = globalTurns.last, last.speaker == turn.speaker,
               turn.start <= last.end + Tuning.sameSpeakerMergeGap {
                globalTurns[globalTurns.count - 1] = GlobalTurn(
                    start: last.start, end: max(last.end, turn.end),
                    localSpeaker: last.localSpeaker,
                    windowIndex: last.windowIndex, speaker: last.speaker)
            } else {
                globalTurns.append(turn)
            }
        }
        guard !globalTurns.isEmpty else { return transcript }
        let speakerTurns = globalTurns.map {
            SpeakerTurn(start: $0.start, end: $0.end, speaker: $0.speaker, name: names[$0.speaker])
        }

        // --- Words: overlap → activity tie-break → nearest fallback ---------
        let words = transcript.words.map { word -> ParakeetWord in
            let mid = (word.start + word.end) / 2
            let overlapping = globalTurns.compactMap { turn -> (turn: GlobalTurn, overlap: Double)? in
                let overlap = min(word.end, turn.end) - max(word.start, turn.start)
                return overlap > 0 ? (turn, overlap) : nil
            }
            let speaker: Int?
            if overlapping.count == 1 {
                speaker = overlapping[0].turn.speaker
            } else if overlapping.count > 1 {
                // Concurrent speech: the locally more active voice wins.
                // Local slots are only valid inside the turn's own
                // segmentation window — clamp the query time into it.
                func activity(_ turn: GlobalTurn) -> Float {
                    let posterior = segmentWindows[turn.windowIndex]
                    let lo = max(turn.start, posterior.startTime)
                    let hi = min(turn.end, posterior.startTime + posterior.duration - 0.001)
                    let query = min(max(mid, lo), max(lo, hi))
                    return posterior.speakerActivity(forLocal: turn.localSpeaker, around: query)
                }
                speaker = overlapping.max(by: { activity($0.turn) < activity($1.turn) })?.turn.speaker
            } else {
                func distance(_ turn: GlobalTurn) -> Double {
                    mid < turn.start ? turn.start - mid : max(0, mid - turn.end)
                }
                let nearest = globalTurns.min(by: { distance($0) < distance($1) })
                speaker = (nearest.map(distance) ?? .infinity) <= Tuning.nearestWordSeconds
                    ? nearest?.speaker : nil
            }
            return word.with(speaker: speaker)
        }

        // Dominant speaker by labelled word time.
        var spoken: [Int: Double] = [:]
        for word in words where word.speaker != nil {
            spoken[word.speaker!, default: 0] += word.end - word.start
        }
        let dominant = spoken.max(by: { $0.value < $1.value })?.key

        return transcript.with(words: words, speaker: .some(dominant), speakerTurns: speakerTurns)
    }

    /// Index of a clean prefix/suffix label change (exactly two values,
    /// ordered a…a b…b with at least two windows on each side), or nil.
    private func cleanLabelChange(_ sequence: [Int]) -> Int? {
        guard sequence.count >= 4, Set(sequence).count == 2 else { return nil }
        guard let cut = sequence.firstIndex(where: { $0 != sequence[0] }) else { return nil }
        guard cut >= 2, sequence.count - cut >= 2 else { return nil }
        let tail = sequence[cut...]
        return tail.allSatisfy { $0 == sequence[cut] } ? cut : nil
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
