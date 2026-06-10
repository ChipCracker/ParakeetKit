//
//  SpeakerDB.swift
//  Persistent named-speaker recognition: enrolled speakers are stored as one
//  JSON file per name (L2-normalized prototype embedding + sample count) and
//  matched by cosine similarity. Functional equivalent of CrispASR's
//  speaker_db (.spkr files), in Swift/Codable instead of the CLI's binary
//  format — profiles are not interchangeable with the CLI.
//
import Foundation

public struct SpeakerProfile: Codable, Sendable, Equatable {
    public let name: String
    /// L2-normalized prototype (running mean of enrolled embeddings, renormalized).
    public var embedding: [Float]
    public var sampleCount: Int

    public init(name: String, embedding: [Float], sampleCount: Int = 1) {
        self.name = name
        self.embedding = embedding
        self.sampleCount = sampleCount
    }
}

/// Thread-safe file-backed store. One `<name>.speaker.json` per speaker in
/// `directory` (created on demand).
public final class SpeakerDB: @unchecked Sendable {
    public static let fileSuffix = ".speaker.json"

    private let lock = NSLock()
    private var profiles: [String: SpeakerProfile] = [:]
    public let directory: URL

    public init(directory: URL) throws {
        self.directory = directory
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        for url in (try? FileManager.default.contentsOfDirectory(at: directory,
                                                                 includingPropertiesForKeys: nil)) ?? []
        where url.lastPathComponent.hasSuffix(Self.fileSuffix) {
            if let data = try? Data(contentsOf: url),
               let profile = try? JSONDecoder().decode(SpeakerProfile.self, from: data) {
                profiles[profile.name] = profile
            }
        }
    }

    public var names: [String] {
        lock.lock(); defer { lock.unlock() }
        return profiles.keys.sorted()
    }

    public var count: Int {
        lock.lock(); defer { lock.unlock() }
        return profiles.count
    }

    /// Best cosine match at or above `threshold`, or nil.
    public func match(_ embedding: [Float], threshold: Float = 0.5) -> (name: String, score: Float)? {
        guard let unit = SpeakerClusterer.normalized(embedding) else { return nil }
        lock.lock(); defer { lock.unlock() }
        var best: (name: String, score: Float)? = nil
        for profile in profiles.values {
            let score = SpeakerClusterer.dot(profile.embedding, unit)
            if score >= threshold, score > (best?.score ?? -2) {
                best = (profile.name, score)
            }
        }
        return best
    }

    /// Enrolls (or refines) a named speaker: the prototype is the running
    /// mean of all enrolled embeddings, renormalized. Persists immediately.
    public func enroll(name: String, embedding: [Float]) throws {
        guard let unit = SpeakerClusterer.normalized(embedding) else {
            throw ParakeetError.invalidEmbedding
        }
        lock.lock(); defer { lock.unlock() }
        var profile: SpeakerProfile
        if var existing = profiles[name] {
            let n = Float(existing.sampleCount)
            var mean = existing.embedding
            for i in 0..<min(mean.count, unit.count) {
                mean[i] = (mean[i] * n + unit[i]) / (n + 1)
            }
            existing.embedding = SpeakerClusterer.normalized(mean) ?? existing.embedding
            existing.sampleCount += 1
            profile = existing
        } else {
            profile = SpeakerProfile(name: name, embedding: unit)
        }
        profiles[name] = profile
        let data = try JSONEncoder().encode(profile)
        try data.write(to: fileURL(for: name), options: .atomic)
    }

    @discardableResult
    public func remove(name: String) throws -> Bool {
        lock.lock(); defer { lock.unlock() }
        guard profiles.removeValue(forKey: name) != nil else { return false }
        try? FileManager.default.removeItem(at: fileURL(for: name))
        return true
    }

    private func fileURL(for name: String) -> URL {
        let safe = name.replacingOccurrences(of: "/", with: "_")
        return directory.appendingPathComponent(safe + Self.fileSuffix)
    }
}
