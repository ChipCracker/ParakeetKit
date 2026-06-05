//
//  VADGating.swift
//  Abstraction so the streaming state machine is VAD-agnostic and testable
//  without the binary (FireRedVAD conforms in the ParakeetKit target).
//
import Foundation

/// A detected speech span (seconds, relative to the passed buffer).
public struct SpeechSegment: Sendable, Equatable {
    public let start: Double
    public let end: Double
    public var duration: Double { end - start }
    public init(start: Double, end: Double) { self.start = start; self.end = end }
}

public protocol VADGating: Sendable {
    func detect(_ samples: [Float], threshold: Float,
                minSpeech: Float, minSilence: Float) async -> [SpeechSegment]
}

/// Treats the whole buffer as one speech segment — used when VAD is disabled or
/// in unit tests.
public struct NoOpVADGate: VADGating {
    public init() {}
    public func detect(_ samples: [Float], threshold: Float,
                       minSpeech: Float, minSilence: Float) async -> [SpeechSegment] {
        guard !samples.isEmpty else { return [] }
        return [SpeechSegment(start: 0, end: Double(samples.count) / 16_000)]
    }
}
