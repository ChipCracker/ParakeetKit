//
//  LiveTranscriber.swift
//  High-level live transcription: wires the microphone + FireRedVAD +
//  ParakeetEngine into a StreamingSession and surfaces its events.
//
//  Out of the box it uses the BUNDLED FireRedVAD model (Bundle.module), so live
//  streaming with VAD gating works with zero app-side configuration.
//
#if os(iOS)
import Foundation
import ParakeetCore

/// Native diarization for a live session. Models come from the catalog
/// (`ParakeetModelCatalog.titanetLarge` / `.pyannoteSegmentation`) via
/// `ModelDownloader`.
public struct LiveDiarization: Sendable {
    /// TitaNet-Large GGUF (speaker embeddings, required).
    public var titanetModelURL: URL
    /// Pyannote segmentation GGUF — enables the word-level final pass. nil = off.
    public var pyannoteModelURL: URL?
    /// Directory of enrolled speaker profiles — enables named recognition. nil = off.
    public var speakerDBDirectory: URL?
    public var options: DiarizationOptions

    public init(titanetModelURL: URL, pyannoteModelURL: URL? = nil,
                speakerDBDirectory: URL? = nil, options: DiarizationOptions = .init()) {
        self.titanetModelURL = titanetModelURL
        self.pyannoteModelURL = pyannoteModelURL
        self.speakerDBDirectory = speakerDBDirectory
        self.options = options
    }
}

public actor LiveTranscriber {
    private let engine: ParakeetEngine
    private let config: StreamingConfig
    private let vadModelURL: URL?
    private let diarization: LiveDiarization?

    private var recorder: AudioRecorder?
    private var session: StreamingSession?
    private var diarizer: Diarizer?

    /// `vadModelURL` defaults to the bundled firered-stream-vad.gguf. Pass `nil`
    /// to disable VAD gating (whole audio treated as speech). `diarization`
    /// enables speaker attribution (`.speaker` events, word-level speakers in
    /// `.finalizedTranscript`).
    public init(engine: ParakeetEngine,
                config: StreamingConfig = .init(),
                vadModelURL: URL? = FireRedVAD.bundledModelURL,
                diarization: LiveDiarization? = nil) {
        self.engine = engine
        self.config = config
        self.vadModelURL = vadModelURL
        self.diarization = diarization
    }

    /// Requests mic permission, starts the AVAudioEngine, feeds blocks into a
    /// StreamingSession and returns its event stream. The stream finishes after
    /// `stop()` runs the final transcribeLong() pass.
    public func start() async throws -> AsyncStream<StreamingEvent> {
        guard await AudioRecorder.requestPermission() else {
            throw ParakeetError.microphonePermissionDenied
        }
        let session = await makeSession()
        self.session = session
        let stream = await session.events()

        let recorder = AudioRecorder()
        self.recorder = recorder
        try recorder.start { samples in
            Task { await session.ingest(samples) }
        }
        return stream
    }

    /// Stops the mic and runs the final long pass (emits `.finalized`, then closes).
    public func stop() async {
        recorder?.stop()
        recorder = nil
        await session?.finish()
        session = nil
    }

    /// Headless: drive the same pipeline from an audio buffer (e.g. a WAV) and
    /// return the committed ("fester") text. Useful for tests / offline runs.
    public func simulate(_ audio: [Float], chunkSamples: Int = 1600) async -> String {
        let session = await makeSession()
        var i = 0
        while i < audio.count {
            let end = min(i + chunkSamples, audio.count)
            await session.drive(Array(audio[i..<end]))
            i = end
        }
        await session.commitRemaining()
        return await session.acceptedText
    }

    /// Enrolls a named speaker from a voice sample (≥ ~1 s) into the
    /// configured speaker DB — future sessions resolve the name via
    /// `.speaker` events / `SpeakerTurn.name`.
    public func enrollSpeaker(name: String, samples: [Float]) async throws {
        guard let diarizer = await currentDiarizer() else {
            throw ParakeetError.modelLoadFailed("diarization not configured")
        }
        try await diarizer.enroll(name: name, samples: samples)
    }

    /// Names a speaker cluster of the current session (e.g. "speaker #2" from
    /// a `.speaker` event): its centroid becomes a persistent profile.
    /// Applies immediately to THIS instance — future `.speaker` events and
    /// the final pass carry the name; other LiveTranscriber instances see the
    /// profile from their next session (the diarizer is cached per instance,
    /// the SpeakerDB loads at init).
    public func enrollSpeaker(name: String, fromClusterID id: Int) async throws {
        guard let diarizer = await currentDiarizer() else {
            throw ParakeetError.modelLoadFailed("diarization not configured")
        }
        try await diarizer.enrollCluster(id: id, name: name)
    }

    private func currentDiarizer() async -> Diarizer? {
        if let diarizer { return diarizer }
        guard let diarization else { return nil }
        diarizer = try? await Diarizer.make(
            titanetModelPath: diarization.titanetModelURL.path,
            pyannoteModelPath: diarization.pyannoteModelURL?.path,
            speakerDBDirectory: diarization.speakerDBDirectory,
            options: diarization.options)
        return diarizer
    }

    private func makeSession() async -> StreamingSession {
        let vad: VADGating
        if let url = vadModelURL, let v = try? await FireRedVAD.make(modelPath: url.path) {
            vad = v
        } else {
            vad = NoOpVADGate()
        }
        let engine = self.engine

        var attributeSpeaker: StreamingSession.SpeakerAttribution? = nil
        var finalizeSpeakers: StreamingSession.SpeakerFinalize? = nil
        if let diarizer = await currentDiarizer() {
            attributeSpeaker = { await diarizer.attribute($0) }
            finalizeSpeakers = { await diarizer.finalize(audio: $0, transcript: $1) }
        }

        return StreamingSession(
            config: config,
            vad: vad,
            transcribe: { await engine.transcribe($0) },
            transcribeLong: { await engine.transcribeLong($0) },
            attributeSpeaker: attributeSpeaker,
            finalizeSpeakers: finalizeSpeakers)
    }
}

#endif
