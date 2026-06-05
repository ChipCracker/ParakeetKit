//
//  StreamingConfig.swift
//  Tunables for live streaming. Defaults match the original StreamingTranscriber.
//
import Foundation

public struct StreamingConfig: Sendable {
    public var sampleRate: Int = 16_000

    // Cadence
    public var vadHopSeconds: Double = 0.3        // run the gate / process every hop
    public var previewStepSeconds: Double = 0.6   // min interval between hypothesis runs

    // Endpointing / commit
    public var endpointSilenceSeconds: Double = 0.8  // commit on a real pause
    public var maxSegmentSeconds: Double = 16        // hard commit cap
    public var speechPadSeconds: Double = 0.3        // tail padding vs. clipped words
    public var preRollSeconds: Double = 0.5          // retained context on silence drop
    public var speakingHangoverSeconds: Double = 0.4 // isSpeaking indicator window
    public var maxFullSeconds: Double = 600          // full-audio ring cap (final pass)

    // VAD detect() parameters
    public var vadThreshold: Float = 0.3
    public var vadMinSpeech: Float = 0.1
    public var vadMinSilence: Float = 0.1

    public init() {}
}
