//
//  ParakeetModelCatalog.swift
//  Registry of downloadable Parakeet models. Ships the 4 built-in quantizations
//  and lets consumers register their own.
//
import Foundation

public final class ParakeetModelCatalog: @unchecked Sendable {
    public static let shared = ParakeetModelCatalog()

    private let lock = NSLock()
    private var order: [String] = []
    private var byID: [String: ParakeetModelSpec] = [:]

    public init(_ specs: [ParakeetModelSpec] = ParakeetModelCatalog.builtins) {
        register(contentsOf: specs)
    }

    public func register(_ spec: ParakeetModelSpec) {
        lock.lock(); defer { lock.unlock() }
        if byID[spec.id] == nil { order.append(spec.id) }
        byID[spec.id] = spec
    }

    public func register(contentsOf specs: [ParakeetModelSpec]) {
        for s in specs { register(s) }
    }

    public func spec(id: String) -> ParakeetModelSpec? {
        lock.lock(); defer { lock.unlock() }
        return byID[id]
    }

    public var all: [ParakeetModelSpec] {
        lock.lock(); defer { lock.unlock() }
        return order.compactMap { byID[$0] }
    }

    public func grouped(by keyPath: KeyPath<ParakeetModelSpec, String>) -> [(key: String, specs: [ParakeetModelSpec])] {
        var keys: [String] = []
        var buckets: [String: [ParakeetModelSpec]] = [:]
        for spec in all {
            let k = spec[keyPath: keyPath]
            if buckets[k] == nil { keys.append(k) }
            buckets[k, default: []].append(spec)
        }
        return keys.map { ($0, buckets[$0] ?? []) }
    }

    // MARK: Built-in Parakeet TDT 0.6B v3 models (cstr/parakeet-tdt-0.6b-v3-GGUF)

    private static let repo = "cstr/parakeet-tdt-0.6b-v3-GGUF"
    private static let family = "Parakeet TDT 0.6B v3"

    public static let q4_K = ParakeetModelSpec.huggingFace(
        id: "parakeet-tdt-0.6b-v3-q4_k", displayName: "q4_K", family: family,
        quantization: .q4_K, repo: repo, fileName: "parakeet-tdt-0.6b-v3-q4_k.gguf",
        approxBytes: 467 * 1_000_000, subtitle: "~467 MB · recommended, fastest")

    public static let q5_0 = ParakeetModelSpec.huggingFace(
        id: "parakeet-tdt-0.6b-v3-q5_0", displayName: "q5_0", family: family,
        quantization: .q5_0, repo: repo, fileName: "parakeet-tdt-0.6b-v3-q5_0.gguf",
        approxBytes: 516 * 1_000_000, subtitle: "~516 MB · slightly more accurate")

    public static let q8_0 = ParakeetModelSpec.huggingFace(
        id: "parakeet-tdt-0.6b-v3-q8_0", displayName: "q8_0", family: family,
        quantization: .q8_0, repo: repo, fileName: "parakeet-tdt-0.6b-v3-q8_0.gguf",
        approxBytes: 711 * 1_000_000, subtitle: "~711 MB · near-lossless")

    public static let f16 = ParakeetModelSpec.huggingFace(
        id: "parakeet-tdt-0.6b-v3-f16", displayName: "F16", family: family,
        quantization: .f16, repo: repo, fileName: "parakeet-tdt-0.6b-v3.gguf",
        approxBytes: 1_260 * 1_000_000, subtitle: "~1.26 GB · full precision")

    public static let builtins: [ParakeetModelSpec] = [q4_K, q5_0, q8_0, f16]
    public static let recommended = q4_K
}
