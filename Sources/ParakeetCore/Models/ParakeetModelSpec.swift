//
//  ParakeetModelSpec.swift
//  Open, extensible description of a downloadable Parakeet GGUF model.
//
import Foundation

public enum ParakeetQuantization: Sendable, Hashable {
    case q4_K, q5_0, q8_0, f16
    case custom(String)

    public var label: String {
        switch self {
        case .q4_K: return "q4_K"
        case .q5_0: return "q5_0"
        case .q8_0: return "q8_0"
        case .f16:  return "F16"
        case .custom(let s): return s
        }
    }
}

public struct ParakeetModelSpec: Sendable, Identifiable, Hashable {
    public let id: String          // "parakeet-tdt-0.6b-v3-q4_k"
    public let displayName: String // "q4_K"
    public let family: String      // "Parakeet TDT 0.6B v3"
    public let quantization: ParakeetQuantization
    public let fileName: String
    public let approxBytes: Int64
    public let subtitle: String?
    public let downloadURL: URL

    public init(id: String, displayName: String, family: String,
                quantization: ParakeetQuantization, fileName: String,
                approxBytes: Int64, subtitle: String? = nil, downloadURL: URL) {
        self.id = id
        self.displayName = displayName
        self.family = family
        self.quantization = quantization
        self.fileName = fileName
        self.approxBytes = approxBytes
        self.subtitle = subtitle
        self.downloadURL = downloadURL
    }

    /// Builds a spec from a Hugging Face repo + filename
    /// (`https://huggingface.co/<repo>/resolve/main/<file>?download=true`).
    public static func huggingFace(id: String, displayName: String, family: String,
                                   quantization: ParakeetQuantization, repo: String,
                                   fileName: String, approxBytes: Int64,
                                   subtitle: String? = nil) -> ParakeetModelSpec {
        let url = URL(string: "https://huggingface.co/\(repo)/resolve/main/\(fileName)?download=true")!
        return ParakeetModelSpec(id: id, displayName: displayName, family: family,
                                 quantization: quantization, fileName: fileName,
                                 approxBytes: approxBytes, subtitle: subtitle, downloadURL: url)
    }

    public static func == (l: ParakeetModelSpec, r: ParakeetModelSpec) -> Bool { l.id == r.id }
    public func hash(into hasher: inout Hasher) { hasher.combine(id) }
}
