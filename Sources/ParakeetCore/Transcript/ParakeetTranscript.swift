//
//  ParakeetTranscript.swift
//  Pure value types for a transcription result (no binary dependency).
//
import Foundation

/// A word with timestamps (from the TDT duration head).
public struct ParakeetWord: Identifiable, Sendable, Hashable {
    public let id: UUID
    public let text: String
    public let start: TimeInterval   // seconds
    public let end: TimeInterval     // seconds
    public let probability: Float

    public init(id: UUID = UUID(), text: String, start: TimeInterval,
                end: TimeInterval, probability: Float) {
        self.id = id
        self.text = text
        self.start = start
        self.end = end
        self.probability = probability
    }
}

/// Result of a transcription.
public struct ParakeetTranscript: Sendable {
    public let text: String
    public let words: [ParakeetWord]
    /// Encoder forward passes for this call (1 per window, n per chunk).
    public let encoderRuns: Int
    /// TDT decode steps (joint evaluations) for this call.
    public let decoderSteps: Int
    /// Duration of processed audio (s) and pure inference time (s).
    public let audioSeconds: Double
    public let processingSeconds: Double

    public init(text: String, words: [ParakeetWord], encoderRuns: Int, decoderSteps: Int,
                audioSeconds: Double, processingSeconds: Double) {
        self.text = text
        self.words = words
        self.encoderRuns = encoderRuns
        self.decoderSteps = decoderSteps
        self.audioSeconds = audioSeconds
        self.processingSeconds = processingSeconds
    }

    /// Real-time factor = processing time / audio duration (< 1 = faster than realtime).
    public var rtf: Double { audioSeconds > 0 ? processingSeconds / audioSeconds : 0 }

    public static let empty = ParakeetTranscript(text: "", words: [], encoderRuns: 0,
                                                 decoderSteps: 0, audioSeconds: 0, processingSeconds: 0)
}
