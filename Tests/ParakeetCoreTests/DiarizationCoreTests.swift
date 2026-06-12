import XCTest
import ParakeetCore

final class DiarizationCoreTests: XCTestCase {

    // Distinct unit vectors in a tiny 4-d space stand in for embeddings.
    private let voiceA: [Float] = [1, 0, 0, 0]
    private let voiceB: [Float] = [0, 1, 0, 0]
    private func noisy(_ v: [Float], _ eps: Float) -> [Float] {
        var out = v
        out[2] += eps
        return out
    }

    // MARK: - SpeakerClusterer

    func testClustererAssignsSameVoiceToSameCluster() {
        var clusterer = SpeakerClusterer(mergeThreshold: 0.5)
        XCTAssertEqual(clusterer.assign(voiceA), 0)
        XCTAssertEqual(clusterer.assign(noisy(voiceA, 0.2)), 0)   // cos ≈ 0.98
        XCTAssertEqual(clusterer.assign(voiceB), 1)               // orthogonal → new
        XCTAssertEqual(clusterer.assign(noisy(voiceB, 0.1)), 1)
        XCTAssertEqual(clusterer.speakerCount, 2)
    }

    func testClustererRespectsMaxSpeakers() {
        var clusterer = SpeakerClusterer(mergeThreshold: 0.9, maxSpeakers: 1)
        XCTAssertEqual(clusterer.assign(voiceA), 0)
        // Orthogonal voice, but the cap forces greedy nearest assignment.
        XCTAssertEqual(clusterer.assign(voiceB), 0)
        XCTAssertEqual(clusterer.speakerCount, 1)
    }

    func testClustererRejectsDegenerateEmbedding() {
        var clusterer = SpeakerClusterer()
        XCTAssertEqual(clusterer.assign([0, 0, 0, 0]), -1)
        XCTAssertEqual(clusterer.speakerCount, 0)
    }

    func testClustererCentroidStaysNormalized() {
        var clusterer = SpeakerClusterer(mergeThreshold: 0.5)
        _ = clusterer.assign(voiceA)
        _ = clusterer.assign(noisy(voiceA, 0.3))
        let norm = clusterer.clusters[0].centroid.reduce(Float(0)) { $0 + $1 * $1 }
        XCTAssertEqual(norm, 1.0, accuracy: 1e-4)
        XCTAssertEqual(clusterer.clusters[0].count, 2)
    }

    // MARK: - SpeakerDB

    private func tempDir() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("speakerdb-\(UUID().uuidString)", isDirectory: true)
    }

    func testSpeakerDBEnrollMatchPersistRoundtrip() throws {
        let dir = tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        let db = try SpeakerDB(directory: dir)
        try db.enroll(name: "christopher", embedding: voiceA)
        XCTAssertEqual(db.match(noisy(voiceA, 0.1))?.name, "christopher")
        XCTAssertNil(db.match(voiceB))                       // below threshold

        // Reload from disk: profile persists.
        let reloaded = try SpeakerDB(directory: dir)
        XCTAssertEqual(reloaded.names, ["christopher"])
        XCTAssertEqual(reloaded.match(voiceA)?.name, "christopher")
    }

    func testSpeakerDBRunningMeanAndRemove() throws {
        let dir = tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        let db = try SpeakerDB(directory: dir)
        try db.enroll(name: "x", embedding: voiceA)
        try db.enroll(name: "x", embedding: noisy(voiceA, 0.2))
        XCTAssertEqual(db.count, 1)
        XCTAssertEqual(db.match(voiceA)?.name, "x")

        XCTAssertTrue(try db.remove(name: "x"))
        XCTAssertEqual(db.count, 0)
        XCTAssertNil(db.match(voiceA))
        XCTAssertEqual(try SpeakerDB(directory: dir).count, 0)  // file gone too
    }

    /// Centroid→profile composition (the enrollCluster semantics): a cluster
    /// centroid built from noisy samples of one voice enrolls as a profile
    /// that matches further samples of that voice — including after reload.
    func testClusterCentroidEnrollsAsMatchingProfile() throws {
        let dir = tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        var clusterer = SpeakerClusterer(mergeThreshold: 0.5)
        XCTAssertEqual(clusterer.assign(voiceA), 0)
        XCTAssertEqual(clusterer.assign(noisy(voiceA, 0.2)), 0)

        let db = try SpeakerDB(directory: dir)
        try db.enroll(name: "alice", embedding: clusterer.clusters[0].centroid)

        XCTAssertEqual(db.match(noisy(voiceA, 0.1))?.name, "alice")
        XCTAssertNil(db.match(voiceB))
        XCTAssertEqual(try SpeakerDB(directory: dir).match(voiceA)?.name, "alice")
    }

    func testSpeakerDBRejectsDegenerateEmbedding() throws {
        let dir = tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let db = try SpeakerDB(directory: dir)
        XCTAssertThrowsError(try db.enroll(name: "bad", embedding: [0, 0, 0]))
    }

    // MARK: - PyannotePosteriors

    /// Builds (frames, 7) log-posteriors where each frame puts ~all mass on
    /// one powerset class.
    private func posteriors(classes: [Int]) -> PyannotePosteriors {
        var logp = [Float](repeating: log(0.001), count: classes.count * 7)
        for (f, k) in classes.enumerated() {
            logp[f * 7 + k] = log(0.994)
        }
        return PyannotePosteriors(logPosteriors: logp, frameCount: classes.count)!
    }

    func testDominantSpeakerAndSilenceGating() {
        // 4 frames spk0 (class 1), 2 frames silence (0), 4 frames spk1 (2).
        let p = posteriors(classes: [1, 1, 1, 1, 0, 0, 2, 2, 2, 2])
        let frame = PyannotePosteriors.frameDuration
        XCTAssertEqual(p.dominantSpeaker(from: 0, to: 4 * frame), 0)
        XCTAssertEqual(p.dominantSpeaker(from: 6 * frame, to: 10 * frame), 1)
        XCTAssertNil(p.dominantSpeaker(from: 4 * frame, to: 6 * frame))   // all silence
        // Overlap class 3 (spk0+1) counts for both, ties broken by first index.
        let overlap = posteriors(classes: [3, 3])
        XCTAssertEqual(overlap.dominantSpeaker(from: 0, to: 2 * frame), 0)
    }

    func testSpeakerTurns() {
        // spk0 ×6, silence ×3, spk2 (class 4) ×6, one-frame blip spk1 dropped.
        let p = posteriors(classes: [1, 1, 1, 1, 1, 1, 0, 0, 0, 4, 4, 4, 4, 4, 4, 2])
        let turns = p.speakerTurns(minTurnSeconds: 2 * PyannotePosteriors.frameDuration)
        XCTAssertEqual(turns.count, 2)
        XCTAssertEqual(turns[0].localSpeaker, 0)
        XCTAssertEqual(turns[1].localSpeaker, 2)
        XCTAssertEqual(turns[0].start, 0, accuracy: 1e-9)
        XCTAssertEqual(turns[0].end, 6 * PyannotePosteriors.frameDuration, accuracy: 1e-6)
        XCTAssertEqual(turns[1].start, 9 * PyannotePosteriors.frameDuration, accuracy: 1e-6)
    }

    func testPosteriorsRejectsShortBuffer() {
        XCTAssertNil(PyannotePosteriors(logPosteriors: [0, 0, 0], frameCount: 1))
    }

    // MARK: - Agglomerative clustering (final pass v2)

    func testAgglomerateSeparatesThreeGroups() {
        let voiceC: [Float] = [0, 0, 1, 0]
        let embeddings = [voiceA, noisy(voiceA, 0.2),
                          voiceB, noisy(voiceB, 0.15),
                          voiceC, noisy(voiceC, 0.2)]
        let labels = SpeakerClusterer.agglomerate(embeddings, stopThreshold: 0.5, maxClusters: 8)
        // Labels are ordered by first member: A=0, B=1, C=2.
        XCTAssertEqual(labels, [0, 0, 1, 1, 2, 2])
    }

    func testAgglomerateHonorsMaxClusters() {
        let voiceC: [Float] = [0, 0, 1, 0]
        // Three mutually orthogonal voices, threshold would keep them apart —
        // the cap forces a merge down to two clusters.
        let labels = SpeakerClusterer.agglomerate([voiceA, voiceB, voiceC],
                                                  stopThreshold: 0.9, maxClusters: 2)
        XCTAssertEqual(Set(labels).count, 2)
        XCTAssertFalse(labels.contains(-1))
    }

    func testAgglomerateEdgeCases() {
        XCTAssertEqual(SpeakerClusterer.agglomerate([], stopThreshold: 0.5, maxClusters: 4), [])
        XCTAssertEqual(SpeakerClusterer.agglomerate([voiceA], stopThreshold: 0.5, maxClusters: 4), [0])
        // Degenerate embedding keeps -1, valid neighbours still cluster.
        let labels = SpeakerClusterer.agglomerate([voiceA, [0, 0, 0, 0], noisy(voiceA, 0.2)],
                                                  stopThreshold: 0.5, maxClusters: 4)
        XCTAssertEqual(labels, [0, -1, 0])
    }

    // MARK: - Overlap-aware segmentation (final pass v2)

    func testPerSpeakerTurnsYieldOverlap() {
        // spk0 alone ×6 (class 1), spk0+spk1 ×4 (class 3), spk1 alone ×6 (2):
        // two OVERLAPPING turns instead of an argmax cut.
        let p = posteriors(classes: [1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2])
        let frame = PyannotePosteriors.frameDuration
        let turns = p.perSpeakerTurns(gapTolerance: 0, minTurnSeconds: 2 * frame)
        XCTAssertEqual(turns.count, 2)
        XCTAssertEqual(turns[0].localSpeaker, 0)
        XCTAssertEqual(turns[0].start, 0, accuracy: 1e-9)
        XCTAssertEqual(turns[0].end, 10 * frame, accuracy: 1e-6)
        XCTAssertEqual(turns[1].localSpeaker, 1)
        XCTAssertEqual(turns[1].start, 6 * frame, accuracy: 1e-6)
        XCTAssertEqual(turns[1].end, 16 * frame, accuracy: 1e-6)
        // Overlap is real: spk0 still runs while spk1 already speaks.
        XCTAssertGreaterThan(turns[0].end, turns[1].start)
    }

    func testPerSpeakerTurnsBridgeShortGaps() {
        // spk0 ×4, one silence frame, spk0 ×4 — bridged into one turn;
        // without tolerance it splits.
        let classes = [1, 1, 1, 1, 0, 1, 1, 1, 1]
        let p = posteriors(classes: classes)
        let frame = PyannotePosteriors.frameDuration
        let bridged = p.perSpeakerTurns(gapTolerance: 2 * frame, minTurnSeconds: 2 * frame)
        XCTAssertEqual(bridged.count, 1)
        XCTAssertEqual(bridged[0].end, 9 * frame, accuracy: 1e-6)
        let split = p.perSpeakerTurns(gapTolerance: 0, minTurnSeconds: 2 * frame)
        XCTAssertEqual(split.count, 2)
    }

    func testPureRangeMasksOverlap() {
        // spk0 ×6 pure, then ×4 overlapped with spk1.
        let p = posteriors(classes: [1, 1, 1, 1, 1, 1, 3, 3, 3, 3])
        let frame = PyannotePosteriors.frameDuration
        let pure = p.pureRange(forLocal: 0, from: 0, to: 10 * frame,
                               minSeconds: 2 * frame)
        XCTAssertNotNil(pure)
        XCTAssertEqual(pure!.start, 0, accuracy: 1e-9)
        XCTAssertEqual(pure!.end, 6 * frame, accuracy: 1e-6)
        // spk1 has no pure stretch in this range at all.
        XCTAssertNil(p.pureRange(forLocal: 1, from: 0, to: 10 * frame,
                                 minSeconds: 2 * frame))
    }

    func testSpeakerActivityWindow() {
        let p = posteriors(classes: [1, 1, 1, 1, 3, 3, 3, 3])
        let frame = PyannotePosteriors.frameDuration
        // In the overlap zone both speakers are highly active.
        let mid = 6 * frame
        XCTAssertGreaterThan(p.speakerActivity(forLocal: 0, around: mid, halfWindow: frame), 0.9)
        XCTAssertGreaterThan(p.speakerActivity(forLocal: 1, around: mid, halfWindow: frame), 0.9)
        // In the pure zone speaker 1 is quiet.
        XCTAssertLessThan(p.speakerActivity(forLocal: 1, around: 2 * frame, halfWindow: frame), 0.1)
    }
}
