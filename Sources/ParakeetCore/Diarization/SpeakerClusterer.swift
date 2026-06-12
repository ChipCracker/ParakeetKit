//
//  SpeakerClusterer.swift
//  Online speaker clustering on L2-normalized embeddings (cosine similarity
//  == dot product). Swift port of the agglomerative idea in CrispASR's
//  crispasr_speaker_cluster, adapted to streaming: embeddings arrive one
//  utterance at a time, so we assign-or-spawn against running centroids
//  instead of re-clustering the full set.
//
//  Threshold rule of thumb for TitaNet-Large (192-d): pairs above ~0.5
//  cosine are usually the same physical speaker, below ~0.4 usually
//  different.
//
import Foundation

public struct SpeakerClusterer: Sendable {
    public struct Cluster: Sendable {
        public internal(set) var centroid: [Float]   // L2-normalized
        public internal(set) var count: Int
    }

    public private(set) var clusters: [Cluster] = []
    public let mergeThreshold: Float
    public let maxSpeakers: Int

    public init(mergeThreshold: Float = 0.5, maxSpeakers: Int = 8) {
        self.mergeThreshold = mergeThreshold
        self.maxSpeakers = max(1, maxSpeakers)
    }

    public var speakerCount: Int { clusters.count }

    /// Assigns an embedding to a speaker index. Joins the most similar
    /// centroid when its cosine similarity reaches `mergeThreshold`, spawns a
    /// new speaker otherwise — but never beyond `maxSpeakers` (then the
    /// nearest centroid wins regardless, greedy). Returns -1 for degenerate
    /// (near-zero) embeddings.
    public mutating func assign(_ embedding: [Float]) -> Int {
        guard let unit = Self.normalized(embedding) else { return -1 }

        var bestIndex = -1
        var bestSim: Float = -2
        for (i, cluster) in clusters.enumerated() {
            let sim = Self.dot(cluster.centroid, unit)
            if sim > bestSim { bestSim = sim; bestIndex = i }
        }

        if bestIndex >= 0, bestSim >= mergeThreshold || clusters.count >= maxSpeakers {
            merge(unit, into: bestIndex)
            return bestIndex
        }
        clusters.append(Cluster(centroid: unit, count: 1))
        return clusters.count - 1
    }

    /// Cosine similarity of an embedding to an existing speaker's centroid.
    public func similarity(of embedding: [Float], toSpeaker index: Int) -> Float? {
        guard clusters.indices.contains(index), let unit = Self.normalized(embedding) else { return nil }
        return Self.dot(clusters[index].centroid, unit)
    }

    public mutating func reset() { clusters.removeAll() }

    /// Offline agglomerative clustering (average linkage over cosine) — the
    /// final pass has ALL embeddings at once, so this beats the
    /// order-sensitive online `assign`. Merges the closest pair while its
    /// similarity reaches `stopThreshold`, and keeps merging regardless while
    /// more than `maxClusters` remain. Returns one label per input (0-based,
    /// in order of each cluster's first member); degenerate (near-zero)
    /// embeddings get -1. Deterministic; n stays small (≤ ~100), so the
    /// naive O(n³) loop is fine.
    public static func agglomerate(_ embeddings: [[Float]],
                                   stopThreshold: Float,
                                   maxClusters: Int) -> [Int] {
        struct Working {
            var centroid: [Float]     // L2-normalized mean
            var sum: [Float]          // un-normalized running sum
            var members: [Int]
        }

        var labels = [Int](repeating: -1, count: embeddings.count)
        var working: [Working] = []
        for (i, embedding) in embeddings.enumerated() {
            guard let unit = normalized(embedding) else { continue }
            working.append(Working(centroid: unit, sum: unit, members: [i]))
        }
        let cap = max(1, maxClusters)

        while working.count > 1 {
            var bestA = 0, bestB = 1
            var bestSim: Float = -2
            for a in 0..<working.count {
                for b in (a + 1)..<working.count {
                    let sim = dot(working[a].centroid, working[b].centroid)
                    if sim > bestSim { bestSim = sim; bestA = a; bestB = b }
                }
            }
            guard bestSim >= stopThreshold || working.count > cap else { break }
            var merged = working[bestA]
            let other = working[bestB]
            for i in 0..<min(merged.sum.count, other.sum.count) {
                merged.sum[i] += other.sum[i]
            }
            merged.centroid = normalized(merged.sum) ?? merged.centroid
            merged.members += other.members
            working.remove(at: bestB)
            working[bestA] = merged
        }

        // Stable labels: clusters numbered by their earliest member.
        let ordered = working.sorted { ($0.members.min() ?? 0) < ($1.members.min() ?? 0) }
        for (label, cluster) in ordered.enumerated() {
            for member in cluster.members { labels[member] = label }
        }
        return labels
    }

    private mutating func merge(_ unit: [Float], into index: Int) {
        var cluster = clusters[index]
        let n = Float(cluster.count)
        var mean = cluster.centroid
        for i in 0..<min(mean.count, unit.count) {
            mean[i] = (mean[i] * n + unit[i]) / (n + 1)
        }
        cluster.centroid = Self.normalized(mean) ?? cluster.centroid
        cluster.count += 1
        clusters[index] = cluster
    }

    // MARK: - Vector helpers (shared by SpeakerDB)

    static func dot(_ a: [Float], _ b: [Float]) -> Float {
        var sum: Float = 0
        for i in 0..<min(a.count, b.count) { sum += a[i] * b[i] }
        return sum
    }

    /// Returns the L2-normalized copy, or nil for near-zero vectors.
    static func normalized(_ v: [Float]) -> [Float]? {
        let norm = Self.dot(v, v).squareRoot()
        guard norm > 1e-6 else { return nil }
        return v.map { $0 / norm }
    }
}
