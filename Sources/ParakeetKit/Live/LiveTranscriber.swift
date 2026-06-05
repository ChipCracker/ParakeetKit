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

public actor LiveTranscriber {
    private let engine: ParakeetEngine
    private let config: StreamingConfig
    private let vadModelURL: URL?

    private var recorder: AudioRecorder?
    private var session: StreamingSession?

    /// `vadModelURL` defaults to the bundled firered-stream-vad.gguf. Pass `nil`
    /// to disable VAD gating (whole audio treated as speech).
    public init(engine: ParakeetEngine,
                config: StreamingConfig = .init(),
                vadModelURL: URL? = FireRedVAD.bundledModelURL) {
        self.engine = engine
        self.config = config
        self.vadModelURL = vadModelURL
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

    private func makeSession() async -> StreamingSession {
        let vad: VADGating
        if let url = vadModelURL, let v = try? await FireRedVAD.make(modelPath: url.path) {
            vad = v
        } else {
            vad = NoOpVADGate()
        }
        let engine = self.engine
        return StreamingSession(
            config: config,
            vad: vad,
            transcribe: { await engine.transcribe($0) },
            transcribeLong: { await engine.transcribeLong($0) })
    }
}

#endif
