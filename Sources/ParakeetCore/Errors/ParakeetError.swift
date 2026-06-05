//
//  ParakeetError.swift
//
import Foundation

public enum ParakeetError: Error, LocalizedError, Sendable {
    case modelLoadFailed(String)
    case downloadFailed(String)
    case microphonePermissionDenied

    public var errorDescription: String? {
        switch self {
        case .modelLoadFailed(let path): return "Failed to load model: \(path)"
        case .downloadFailed(let msg):   return "Model download failed: \(msg)"
        case .microphonePermissionDenied: return "Microphone permission denied."
        }
    }
}

public enum VADError: Error, LocalizedError, Sendable {
    case loadFailed(String)
    case bundledModelMissing

    public var errorDescription: String? {
        switch self {
        case .loadFailed(let path):  return "Failed to load VAD model: \(path)"
        case .bundledModelMissing:   return "Bundled VAD model (firered-stream-vad.gguf) is missing."
        }
    }
}

public enum AudioLoadError: Error, LocalizedError, Sendable {
    case cannotOpen(String)
    case conversionFailed

    public var errorDescription: String? {
        switch self {
        case .cannotOpen(let name): return "Could not open audio file: \(name)"
        case .conversionFailed:     return "Audio conversion failed."
        }
    }
}
