//
//  StreamingSession.swift
//  Live transcription state machine (hypothesis → commit), decoupled from
//  SwiftUI and from the binary. Driven by an injected `transcribe` closure and a
//  `VADGating`, so it is fully unit-testable without the parakeet binary.
//
//  Pipeline: ingested 16 kHz mono blocks → (VAD gate) → ASR. While speaking, a
//  rolling PREVIEW ("hypothesis"/"hyp") is produced every `previewStepSeconds`;
//  at a real pause (or `maxSegmentSeconds`) the window is COMMITTED ("fester"
//  text) and its encoder/decoder runs accumulate into `StreamingStats`.
//
import Foundation

public actor StreamingSession {
    public typealias Transcribe = @Sendable ([Float]) async -> ParakeetTranscript

    private let config: StreamingConfig
    private let vad: VADGating
    private let transcribe: Transcribe
    private let transcribeLong: Transcribe

    private var continuation: AsyncStream<StreamingEvent>.Continuation?

    private var committedText = ""
    private var segment: [Float] = []
    private var fullAudio: [Float] = []
    private var samplesSinceTick = 0
    private var samplesSincePreview = 0
    private var processing = false

    // Preview window cap (previewWindowSeconds): previews transcribe only the
    // tail window; words that scrolled out are frozen into a display prefix.
    // All sample indices are relative to the CURRENT segment base —
    // `segmentGeneration` bumps whenever the base shifts (commit/trim/reset),
    // invalidating anything remembered across an `await`.
    private var hypPrefixText = ""
    private var hypPrefixEndSample = 0
    private var lastPreview: (start: Int, end: Int, generation: Int, result: ParakeetTranscript)?
    private var segmentGeneration = 0

    public private(set) var hypothesisText = ""
    public private(set) var isSpeaking = false
    public private(set) var stats = StreamingStats()
    /// Committed ("fester") transcript accumulated so far.
    public var acceptedText: String { committedText }

    public init(config: StreamingConfig = .init(),
                vad: VADGating = NoOpVADGate(),
                transcribe: @escaping Transcribe,
                transcribeLong: Transcribe? = nil) {
        self.config = config
        self.vad = vad
        self.transcribe = transcribe
        self.transcribeLong = transcribeLong ?? transcribe
    }

    /// The hot event stream. Call once; events flow until `finish()`.
    public func events() -> AsyncStream<StreamingEvent> {
        let (stream, continuation) = AsyncStream<StreamingEvent>.makeStream()
        self.continuation = continuation
        return stream
    }

    // MARK: - Audio input

    /// Feed 16 kHz mono Float32 blocks (mic). Non-blocking: heavy work runs in a
    /// child task guarded by `processing`.
    public func ingest(_ samples: [Float]) {
        if let snapshot = accumulate(samples) {
            Task { await self.process(snapshot) }
        }
    }

    /// Deterministic variant for tests / file simulation: awaits processing inline.
    public func drive(_ samples: [Float]) async {
        if let snapshot = accumulate(samples) {
            await process(snapshot)
        }
    }

    private func accumulate(_ samples: [Float]) -> [Float]? {
        continuation?.yield(.level(min(1, Self.rms(samples) * 6)))
        segment.append(contentsOf: samples)
        fullAudio.append(contentsOf: samples)
        let maxFull = Int(config.maxFullSeconds * Double(config.sampleRate))
        if fullAudio.count > maxFull { fullAudio.removeFirst(fullAudio.count - maxFull) }

        samplesSinceTick += samples.count
        samplesSincePreview += samples.count
        let hop = Int(config.vadHopSeconds * Double(config.sampleRate))
        guard samplesSinceTick >= hop, !processing, !segment.isEmpty else { return nil }
        samplesSinceTick = 0
        processing = true
        return segment
    }

    // MARK: - VAD gating + endpointing

    private func process(_ snapshot: [Float]) async {
        defer { processing = false }
        let sr = Double(config.sampleRate)
        let bufDuration = Double(snapshot.count) / sr

        let segs = await vad.detect(snapshot, threshold: config.vadThreshold,
                                    minSpeech: config.vadMinSpeech, minSilence: config.vadMinSilence)

        guard let lastSpeechEnd = segs.last?.end else {
            isSpeaking = false
            continuation?.yield(.speaking(false))
            hypothesisText = ""                       // silence → no running hypothesis
            continuation?.yield(.hypothesis(""))
            invalidatePreviewState()
            let keep = Int(config.preRollSeconds * sr)
            if snapshot.count > keep {
                segment.removeFirst(min(snapshot.count - keep, segment.count))
                segmentGeneration += 1
            }
            return
        }

        let trailingSilence = bufDuration - lastSpeechEnd
        // Speech part incl. tail padding (against clipped words); everything from
        // index 0 is kept — gaps between speech segments are transcribed too.
        let padEnd = min(lastSpeechEnd + config.speechPadSeconds, bufDuration)
        let speechEndSample = min(Int(padEnd * sr), snapshot.count)
        isSpeaking = trailingSilence < config.speakingHangoverSeconds
        continuation?.yield(.speaking(isSpeaking))
        // Commit only at a real pause → whole utterances, better ASR quality.
        let endpoint = trailingSilence >= config.endpointSilenceSeconds || bufDuration >= config.maxSegmentSeconds

        if endpoint {
            // Commits always transcribe the FULL speech window — the preview
            // cap never touches the committed/final text. If the last preview
            // already saw exactly this window (start 0 = no frozen prefix,
            // same end, base unshifted), its result IS the commit result
            // (greedy decode is deterministic) — skip the duplicate run.
            let result: ParakeetTranscript
            if config.reuseLastPreviewOnCommit,
               let last = lastPreview, last.generation == segmentGeneration,
               last.start == 0, last.end == speechEndSample {
                result = last.result
            } else {
                let speech = Array(snapshot[0..<speechEndSample])
                result = await transcribe(speech)
            }
            let text = result.text.trimmingCharacters(in: .whitespacesAndNewlines)
            if !text.isEmpty {
                committedText += committedText.isEmpty ? text : " " + text
            }
            stats.encoderRuns += result.encoderRuns
            stats.decoderSteps += result.decoderSteps
            stats.audioSeconds += result.audioSeconds
            stats.processingSeconds += result.processingSeconds
            segment.removeFirst(min(speechEndSample, segment.count))
            segmentGeneration += 1
            invalidatePreviewState()
            samplesSincePreview = 0
            hypothesisText = ""
            if !text.isEmpty { continuation?.yield(.committed(segment: text, full: committedText)) }
            continuation?.yield(.stats(stats))
            continuation?.yield(.hypothesis(""))
        } else {
            // Adaptive cadence: long segments preview less often — the longer
            // the window, the less a 0.6 s refresh adds for the reader.
            let step = bufDuration >= config.previewSlowAfterSeconds
                ? config.previewStepSlowSeconds : config.previewStepSeconds
            guard Double(samplesSincePreview) >= step * sr else { return }
            samplesSincePreview = 0
            let generation = segmentGeneration
            let windowStart = previewWindowStart(speechEndSample: speechEndSample, sr: sr,
                                                 generation: generation)
            let speech = Array(snapshot[windowStart..<speechEndSample])
            let result = await transcribe(speech)
            // The segment base may have shifted while we were transcribing
            // (commitRemaining/finish/reset interleaved) — drop stale results.
            guard segmentGeneration == generation else { return }
            lastPreview = (windowStart, speechEndSample, generation, result)
            let windowText = result.text.trimmingCharacters(in: .whitespacesAndNewlines)
            hypothesisText = [hypPrefixText, windowText].filter { !$0.isEmpty }
                .joined(separator: " ")
            continuation?.yield(.hypothesis(hypothesisText))
        }
    }

    /// Window start for the next preview. Words of the last preview that ended
    /// before the cap boundary are promoted into the frozen display prefix, and
    /// the window snaps to that word boundary (no mid-word cuts, no overlap
    /// between prefix and window text). Transcribers that return no word
    /// timestamps keep the full window (old behaviour).
    private func previewWindowStart(speechEndSample: Int, sr: Double, generation: Int) -> Int {
        guard config.previewWindowSeconds > 0 else { return 0 }
        let desired = speechEndSample - Int(config.previewWindowSeconds * sr)
        if desired > hypPrefixEndSample,
           let last = lastPreview, last.generation == generation {
            var promoted: [String] = []
            var promotedEnd = hypPrefixEndSample
            for word in last.result.words {
                let wordEnd = last.start + Int(word.end * sr)
                guard wordEnd <= desired else { break }   // words are time-ordered
                promoted.append(word.text)
                promotedEnd = max(promotedEnd, wordEnd)
            }
            if !promoted.isEmpty {
                hypPrefixText = ([hypPrefixText] + promoted).filter { !$0.isEmpty }
                    .joined(separator: " ")
                hypPrefixEndSample = promotedEnd
            }
        }
        return min(hypPrefixEndSample, speechEndSample)
    }

    private func invalidatePreviewState() {
        hypPrefixText = ""
        hypPrefixEndSample = 0
        lastPreview = nil
    }

    // MARK: - Finalisation

    /// Transcribes any leftover buffered speech and commits it (commit-based
    /// finalisation, e.g. after a file simulation). Does not close the stream.
    public func commitRemaining() async {
        guard !segment.isEmpty else { return }
        let result = await transcribe(segment)
        let text = result.text.trimmingCharacters(in: .whitespacesAndNewlines)
        if !text.isEmpty {
            committedText += committedText.isEmpty ? text : " " + text
        }
        stats.encoderRuns += result.encoderRuns
        stats.decoderSteps += result.decoderSteps
        stats.audioSeconds += result.audioSeconds
        stats.processingSeconds += result.processingSeconds
        segment.removeAll()
        segmentGeneration += 1
        invalidatePreviewState()
        if !text.isEmpty { continuation?.yield(.committed(segment: text, full: committedText)) }
        continuation?.yield(.stats(stats))
    }

    /// Stops the session: runs the final `transcribeLong()` pass over the full
    /// audio, emits `.finalized`, then closes the event stream.
    public func finish() async {
        let audio = fullAudio
        if !audio.isEmpty {
            let result = await transcribeLong(audio)
            continuation?.yield(.finalized(result.text.trimmingCharacters(in: .whitespacesAndNewlines)))
        }
        continuation?.finish()
    }

    /// Clears all buffers + counters (re-arm for a fresh utterance run).
    public func reset() {
        committedText = ""; hypothesisText = ""
        segment.removeAll(keepingCapacity: true)
        fullAudio.removeAll(keepingCapacity: true)
        samplesSinceTick = 0; samplesSincePreview = 0; processing = false
        isSpeaking = false; stats = StreamingStats()
        segmentGeneration += 1
        invalidatePreviewState()
    }

    private static func rms(_ samples: [Float]) -> Float {
        guard !samples.isEmpty else { return 0 }
        var sum: Float = 0
        for s in samples { sum += s * s }
        return (sum / Float(samples.count)).squareRoot()
    }
}
