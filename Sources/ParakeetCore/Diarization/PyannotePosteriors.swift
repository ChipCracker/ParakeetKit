//
//  PyannotePosteriors.swift
//  Pure-logic evaluation of pyannote-segmentation-3.0 outputs: the C runtime
//  (pyannote_seg_run) returns row-major (T, 7) log-softmax posteriors over
//  the powerset classes; this maps them onto per-speaker activity, dominant
//  speakers per time range, and speaker turns. Swift port of CrispASR's
//  assign_speakers_from_log_posteriors scoring.
//
//  Class semantics (powerset of ≤2 concurrent speakers out of 3 locals):
//    0 = silence, 1 = spk0, 2 = spk1, 3 = spk0+1, 4 = spk2,
//    5 = spk0+2, 6 = spk1+2
//
import Foundation

public struct PyannotePosteriors: Sendable {
    /// pyannote-segmentation-3.0 emits one frame per 270 samples @ 16 kHz.
    public static let frameDuration: Double = 270.0 / 16_000.0
    public static let localSpeakerCount = 3

    /// Row-major (frameCount, 7) log-softmax values.
    public let logPosteriors: [Float]
    public let frameCount: Int
    /// Absolute start time of frame 0 (seconds) within the original audio.
    public let startTime: Double

    public init?(logPosteriors: [Float], frameCount: Int, startTime: Double = 0) {
        guard frameCount > 0, logPosteriors.count >= frameCount * 7 else { return nil }
        self.logPosteriors = logPosteriors
        self.frameCount = frameCount
        self.startTime = startTime
    }

    public var duration: Double { Double(frameCount) * Self.frameDuration }

    /// (silence, [spk0, spk1, spk2]) probabilities for one frame.
    /// Speaker activity sums the powerset classes that contain the speaker.
    public func activity(atFrame f: Int) -> (silence: Float, speakers: [Float]) {
        let base = f * 7
        var p = [Float](repeating: 0, count: 7)
        for k in 0..<7 { p[k] = expf(logPosteriors[base + k]) }
        return (p[0], [p[1] + p[3] + p[5],   // spk0
                       p[2] + p[3] + p[6],   // spk1
                       p[4] + p[5] + p[6]])  // spk2
    }

    /// Dominant local speaker (0…2) in [from, to) seconds (absolute), or nil
    /// when every frame is silence-gated. Frames with
    /// p(silence) > silenceThreshold do not vote.
    public func dominantSpeaker(from: Double, to: Double, silenceThreshold: Float = 0.5) -> Int? {
        let f0 = max(0, Int((from - startTime) / Self.frameDuration))
        let f1 = min(frameCount, Int(ceil((to - startTime) / Self.frameDuration)))
        guard f1 > f0 else { return nil }
        var score = [Float](repeating: 0, count: Self.localSpeakerCount)
        var voted = false
        for f in f0..<f1 {
            let (silence, speakers) = activity(atFrame: f)
            guard silence <= silenceThreshold else { continue }
            voted = true
            for s in 0..<Self.localSpeakerCount { score[s] += speakers[s] }
        }
        guard voted, let maxScore = score.max(), maxScore > 0 else { return nil }
        return score.firstIndex(of: maxScore)
    }

    /// Overlap-aware speaker turns: one independent activity track per local
    /// speaker, binarized with hysteresis (on at `onThreshold`, off below
    /// `offThreshold`), gaps shorter than `gapTolerance` bridged, runs
    /// shorter than `minTurnSeconds` dropped. Unlike `speakerTurns()` the
    /// results MAY overlap in time — that is the point: concurrent speech
    /// yields one turn per active speaker.
    public func perSpeakerTurns(onThreshold: Float = 0.5,
                                offThreshold: Float = 0.35,
                                gapTolerance: Double = 0.25,
                                minTurnSeconds: Double = 0.3)
        -> [(start: Double, end: Double, localSpeaker: Int)] {
        let gapFrames = Int(gapTolerance / Self.frameDuration)
        var turns: [(start: Double, end: Double, localSpeaker: Int)] = []

        for speaker in 0..<Self.localSpeakerCount {
            var runs: [(start: Int, end: Int)] = []   // [start, end) in frames
            var runStart: Int? = nil
            var active = false
            for f in 0..<frameCount {
                let level = activity(atFrame: f).speakers[speaker]
                if active {
                    if level < offThreshold {
                        runs.append((runStart!, f))
                        runStart = nil
                        active = false
                    }
                } else if level >= onThreshold {
                    runStart = f
                    active = true
                }
            }
            if let start = runStart { runs.append((start, frameCount)) }

            // Bridge short gaps, then drop short runs.
            var merged: [(start: Int, end: Int)] = []
            for run in runs {
                if let last = merged.last, run.start - last.end <= gapFrames {
                    merged[merged.count - 1].end = run.end
                } else {
                    merged.append(run)
                }
            }
            for run in merged {
                let start = startTime + Double(run.start) * Self.frameDuration
                let end = startTime + Double(run.end) * Self.frameDuration
                if end - start >= minTurnSeconds {
                    turns.append((start, end, speaker))
                }
            }
        }
        return turns.sorted { $0.start < $1.start }
    }

    /// Longest sub-interval of [from, to) in which ONLY `localSpeaker` is
    /// active (its activity ≥ `offThreshold`, every other speaker below it) —
    /// purity masking for speaker embeddings over overlapped turns. Returns
    /// nil when no pure stretch reaches `minSeconds`.
    public func pureRange(forLocal localSpeaker: Int, from: Double, to: Double,
                          offThreshold: Float = 0.35,
                          minSeconds: Double = 0.6) -> (start: Double, end: Double)? {
        let f0 = max(0, Int((from - startTime) / Self.frameDuration))
        let f1 = min(frameCount, Int(ceil((to - startTime) / Self.frameDuration)))
        guard f1 > f0 else { return nil }

        var best: (start: Int, end: Int)? = nil
        var runStart: Int? = nil
        func close(_ end: Int) {
            guard let start = runStart else { return }
            if end - start > ((best?.end ?? 0) - (best?.start ?? 0)) {
                best = (start, end)
            }
            runStart = nil
        }
        for f in f0..<f1 {
            let speakers = activity(atFrame: f).speakers
            let pure = speakers[localSpeaker] >= offThreshold
                && speakers.enumerated().allSatisfy { $0.offset == localSpeaker || $0.element < offThreshold }
            if pure {
                if runStart == nil { runStart = f }
            } else {
                close(f)
            }
        }
        close(f1)

        guard let range = best else { return nil }
        let start = startTime + Double(range.start) * Self.frameDuration
        let end = startTime + Double(range.end) * Self.frameDuration
        return end - start >= minSeconds ? (start, end) : nil
    }

    /// Mean activity of a local speaker in a small window around `time`
    /// (absolute seconds) — tie-breaker when a word overlaps several turns.
    public func speakerActivity(forLocal localSpeaker: Int, around time: Double,
                                halfWindow: Double = 0.1) -> Float {
        let f0 = max(0, Int((time - halfWindow - startTime) / Self.frameDuration))
        let f1 = min(frameCount, Int(ceil((time + halfWindow - startTime) / Self.frameDuration)))
        guard f1 > f0 else { return 0 }
        var sum: Float = 0
        for f in f0..<f1 { sum += activity(atFrame: f).speakers[localSpeaker] }
        return sum / Float(f1 - f0)
    }

    /// Contiguous single-speaker regions (local IDs) over the whole buffer:
    /// per frame the dominant non-silence speaker, merged into runs, dropping
    /// runs shorter than `minTurnSeconds`. Used by the final pass to slice
    /// audio for per-turn speaker embeddings.
    public func speakerTurns(silenceThreshold: Float = 0.5,
                             minTurnSeconds: Double = 0.4) -> [(start: Double, end: Double, localSpeaker: Int)] {
        var turns: [(start: Double, end: Double, localSpeaker: Int)] = []
        var current: (start: Int, speaker: Int)? = nil

        func close(_ endFrame: Int) {
            guard let c = current else { return }
            let start = startTime + Double(c.start) * Self.frameDuration
            let end = startTime + Double(endFrame) * Self.frameDuration
            if end - start >= minTurnSeconds {
                turns.append((start, end, c.speaker))
            }
            current = nil
        }

        for f in 0..<frameCount {
            let (silence, speakers) = activity(atFrame: f)
            let dominant: Int? = silence > silenceThreshold
                ? nil
                : speakers.firstIndex(of: speakers.max() ?? 0)
            switch (current, dominant) {
            case (nil, nil):
                break
            case (nil, .some(let s)):
                current = (f, s)
            case (.some, nil):
                close(f)
            case (.some(let c), .some(let s)) where c.speaker != s:
                close(f)
                current = (f, s)
            default:
                break
            }
        }
        close(frameCount)
        return turns
    }
}
