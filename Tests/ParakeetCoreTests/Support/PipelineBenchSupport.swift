//
//  PipelineBenchSupport.swift
//  Deterministic building blocks for the macOS streaming-pipeline benchmark:
//  a counting transcriber, an energy VAD, and scripted speech/silence audio.
//  No binary dependency — drives StreamingSession pure-logic.
//
import Foundation
import ParakeetCore

/// Counts every transcribe call and the audio fed into it (previews + commits).
actor CallRecorder {
    private(set) var calls = 0
    private(set) var samplesTotal = 0

    func record(sampleCount: Int) {
        calls += 1
        samplesTotal += sampleCount
    }

    var audioSeconds: Double { Double(samplesTotal) / 16_000 }
}

/// Deterministic energy VAD: a 10 ms hop is speech when its peak exceeds 0.05
/// (scripted speech uses 0.1 amplitude, silence 0.0 — the `threshold` parameter
/// is a probability for real VADs and is intentionally ignored here).
/// Adjacent segments closer than `minSilence` are merged; segments shorter
/// than `minSpeech` are dropped.
struct EnergyVAD: VADGating {
    func detect(_ samples: [Float], threshold: Float,
                minSpeech: Float, minSilence: Float) async -> [SpeechSegment] {
        let sr = 16_000.0
        let hop = 160 // 10 ms
        guard !samples.isEmpty else { return [] }

        var raw: [(start: Double, end: Double)] = []
        var segStart: Int? = nil
        var i = 0
        while i < samples.count {
            let end = min(i + hop, samples.count)
            var peak: Float = 0
            for j in i..<end { peak = max(peak, abs(samples[j])) }
            if peak > 0.05 {
                if segStart == nil { segStart = i }
            } else if let s = segStart {
                raw.append((Double(s) / sr, Double(i) / sr))
                segStart = nil
            }
            i = end
        }
        if let s = segStart { raw.append((Double(s) / sr, Double(samples.count) / sr)) }

        // Merge gaps < minSilence, then drop segments < minSpeech.
        var merged: [(start: Double, end: Double)] = []
        for seg in raw {
            if var last = merged.last, seg.start - last.end < Double(minSilence) {
                last.end = seg.end
                merged[merged.count - 1] = last
            } else {
                merged.append(seg)
            }
        }
        return merged.filter { $0.end - $0.start >= Double(minSpeech) }
            .map { SpeechSegment(start: $0.start, end: $0.end) }
    }
}

/// Builds audio from a speech/silence script: speech = constant 0.1 amplitude,
/// silence = 0.0. The speak/pause plan lives in the audio itself, so the
/// pipeline (VAD, endpointing) behaves exactly as in production.
func scriptedAudio(_ parts: [(speech: Double, silence: Double)], sr: Int = 16_000) -> [Float] {
    var out: [Float] = []
    for part in parts {
        out.append(contentsOf: [Float](repeating: 0.1, count: Int(part.speech * Double(sr))))
        out.append(contentsOf: [Float](repeating: 0.0, count: Int(part.silence * Double(sr))))
    }
    return out
}

/// Deterministic fake transcriber: emits "w0 w1 … wN" (one word per second of
/// window length) with window-relative word timestamps (start: i, end: i+0.9),
/// mirroring the real engine's t_offset=0 behaviour. Windows that contain no
/// speech energy return an empty transcript (like the real model on silence).
/// encoderRuns=1 per call so `stats.encoderRuns` equals the commit-call count.
func countingFakeTranscribe(recorder: CallRecorder) -> StreamingSession.Transcribe {
    { samples in
        await recorder.record(sampleCount: samples.count)
        let hasSpeech = samples.contains { abs($0) > 0.05 }
        guard hasSpeech else {
            return ParakeetTranscript(text: "", words: [], encoderRuns: 1, decoderSteps: 0,
                                      audioSeconds: Double(samples.count) / 16_000,
                                      processingSeconds: 0)
        }
        let seconds = Double(samples.count) / 16_000
        let n = max(1, Int(seconds))
        var words: [ParakeetWord] = []
        words.reserveCapacity(n)
        for i in 0..<n {
            words.append(ParakeetWord(text: "w\(i)", start: Double(i),
                                      end: Double(i) + 0.9, probability: 1))
        }
        return ParakeetTranscript(text: words.map(\.text).joined(separator: " "),
                                  words: words, encoderRuns: 1, decoderSteps: n,
                                  audioSeconds: seconds, processingSeconds: 0)
    }
}

/// One benchmark scenario's pipeline cost profile.
struct PipelineBenchResult: Codable {
    let scenario: String
    let transcribeCalls: Int
    let previewCalls: Int
    let commitCalls: Int
    let previewAudioSeconds: Double
    let commitAudioSeconds: Double
    let ingestedAudioSeconds: Double
    let committedText: String

    var markdownRow: String {
        String(format: "| %@ | %d | %d | %d | %.1f | %.1f | %.1f |",
               scenario, transcribeCalls, previewCalls, commitCalls,
               previewAudioSeconds, commitAudioSeconds, ingestedAudioSeconds)
    }

    static let markdownHeader = """
    | scenario | calls | preview | commit | previewAudio s | commitAudio s | ingested s |
    |---|---|---|---|---|---|---|
    """
}

enum BenchOutput {
    /// Writes pretty JSON to $PARAKEET_BENCH_OUT/<name>.json when the env var
    /// is set (benchmark.sh sets it); always prints to the console.
    static func write<T: Encodable>(_ value: T, name: String) {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        guard let data = try? encoder.encode(value),
              let json = String(data: data, encoding: .utf8) else { return }
        print("[bench] \(name):\n\(json)")
        guard let dir = ProcessInfo.processInfo.environment["PARAKEET_BENCH_OUT"] else { return }
        let dirURL = URL(fileURLWithPath: dir, isDirectory: true)
        try? FileManager.default.createDirectory(at: dirURL, withIntermediateDirectories: true)
        try? data.write(to: dirURL.appendingPathComponent("\(name).json"))
    }
}
