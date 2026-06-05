//
//  ParakeetEngine.swift
//  Swift wrapper around the parakeet.cpp (CrispASR) C API (module `CParakeet`).
//
//  Inference is blocking, so the engine is an `actor` that serialises all calls
//  and keeps them off the main thread.
//
#if os(iOS)
import Foundation
import ParakeetCore
import CParakeet

public actor ParakeetEngine {
    /// Sensible GPU default per environment: Metal on real devices, CPU in the
    /// iOS simulator (whose software Metal driver can't reliably allocate ggml
    /// buffers).
    public static var preferredUseGPU: Bool {
        #if targetEnvironment(simulator)
        return false
        #else
        return true
        #endif
    }

    private let ctx: OpaquePointer

    /// Loads a GGUF model. Runs synchronously (several seconds) — call via
    /// `make(...)` to stay off-main.
    public init(modelPath: String, useGPU: Bool = true, threads: Int? = nil) throws {
        var params = parakeet_context_default_params()
        params.use_gpu = useGPU
        params.verbosity = 1
        if let threads {
            params.n_threads = Int32(threads)
        }
        guard let ctx = parakeet_init_from_file(modelPath, params) else {
            throw ParakeetError.modelLoadFailed(modelPath)
        }
        self.ctx = ctx
    }

    deinit {
        parakeet_free(ctx)
    }

    /// Creates the engine on a background thread (model loading blocks).
    public static func make(modelPath: String, useGPU: Bool = true,
                            threads: Int? = nil) async throws -> ParakeetEngine {
        try await Task.detached(priority: .userInitiated) {
            try ParakeetEngine(modelPath: modelPath, useGPU: useGPU, threads: threads)
        }.value
    }

    /// Convenience: load a downloaded `ParakeetModelSpec`.
    public static func make(spec: ParakeetModelSpec, downloadedAt url: URL,
                            useGPU: Bool = ParakeetEngine.preferredUseGPU,
                            threads: Int? = nil) async throws -> ParakeetEngine {
        try await make(modelPath: url.path, useGPU: useGPU, threads: threads)
    }

    public var sampleRate: Int { Int(parakeet_sample_rate(ctx)) }

    /// Transcribes a complete sample buffer (16 kHz mono Float32).
    public func transcribe(_ samples: [Float]) -> ParakeetTranscript {
        guard !samples.isEmpty else { return .empty }
        let audioSeconds = Double(samples.count) / 16_000.0
        return samples.withUnsafeBufferPointer { buf -> ParakeetTranscript in
            let t0 = Date()
            guard let res = parakeet_transcribe_ex(ctx, buf.baseAddress, Int32(buf.count), 0) else {
                return .empty
            }
            let processing = Date().timeIntervalSince(t0)
            defer { parakeet_result_free(res) }
            return Self.convert(res.pointee,
                                encoderRuns: Int(parakeet_last_encoder_runs(ctx)),
                                decoderSteps: Int(parakeet_last_decoder_steps(ctx)),
                                audioSeconds: audioSeconds,
                                processingSeconds: processing)
        }
    }

    /// Transcribes long audio in overlapping windows (TDT chunking).
    public func transcribeLong(_ samples: [Float], chunkSeconds: Int = 20,
                               overlapSeconds: Int = 2) -> ParakeetTranscript {
        guard !samples.isEmpty else { return .empty }
        let audioSeconds = Double(samples.count) / 16_000.0
        return samples.withUnsafeBufferPointer { buf -> ParakeetTranscript in
            let t0 = Date()
            guard let res = parakeet_transcribe_chunked(ctx, buf.baseAddress, Int32(buf.count), 0,
                                                        Int32(chunkSeconds), Int32(overlapSeconds)) else {
                return .empty
            }
            let processing = Date().timeIntervalSince(t0)
            defer { parakeet_result_free(res) }
            return Self.convert(res.pointee,
                                encoderRuns: Int(parakeet_last_encoder_runs(ctx)),
                                decoderSteps: Int(parakeet_last_decoder_steps(ctx)),
                                audioSeconds: audioSeconds,
                                processingSeconds: processing)
        }
    }

    // MARK: - C → Swift

    private static func convert(_ r: parakeet_result, encoderRuns: Int, decoderSteps: Int,
                                audioSeconds: Double, processingSeconds: Double) -> ParakeetTranscript {
        let text = r.text.map { String(cString: $0) } ?? ""
        var words: [ParakeetWord] = []
        if let wptr = r.words, r.n_words > 0 {
            words.reserveCapacity(Int(r.n_words))
            for i in 0..<Int(r.n_words) {
                let w = wptr[i]
                let str = Self.fixedCString(w.text)
                words.append(ParakeetWord(text: str,
                                          start: Double(w.t0) / 100.0,
                                          end: Double(w.t1) / 100.0,
                                          probability: w.p))
            }
        }
        return ParakeetTranscript(text: text.trimmingCharacters(in: .whitespacesAndNewlines),
                                  words: words,
                                  encoderRuns: encoderRuns,
                                  decoderSteps: decoderSteps,
                                  audioSeconds: audioSeconds,
                                  processingSeconds: processingSeconds)
    }

    /// Converts a fixed C `char[]` tuple into a Swift string.
    private static func fixedCString<T>(_ tuple: T) -> String {
        withUnsafePointer(to: tuple) { ptr in
            ptr.withMemoryRebound(to: CChar.self, capacity: MemoryLayout<T>.size) {
                String(cString: $0)
            }
        }
    }
}

#endif
