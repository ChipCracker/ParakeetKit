//
//  StreamingEvent.swift
//
import Foundation

/// Cumulative metrics over COMMITTED segments (hypotheses do not accumulate).
public struct StreamingStats: Sendable, Equatable {
    public var encoderRuns: Int = 0
    public var decoderSteps: Int = 0
    public var audioSeconds: Double = 0
    public var processingSeconds: Double = 0
    public var rtf: Double { audioSeconds > 0 ? processingSeconds / audioSeconds : 0 }
    public init() {}
}

/// Events emitted by a streaming session.
public enum StreamingEvent: Sendable {
    /// Grey, in-flight preview of the current window ("hyp" text). Not committed.
    case hypothesis(String)
    /// Finalised ("fester") text: the newly committed piece + the full accepted text.
    case committed(segment: String, full: String)
    /// VAD speaking indicator.
    case speaking(Bool)
    /// RMS audio level 0…1 (for a level meter / waveform).
    case level(Float)
    /// Cumulative committed-run counters.
    case stats(StreamingStats)
    /// Final transcribeLong() pass over the whole session, emitted on finish().
    case finalized(String)
}
