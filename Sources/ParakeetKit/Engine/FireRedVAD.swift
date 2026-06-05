//
//  FireRedVAD.swift
//  Swift wrapper around the FireRedVAD (DFSMN) C API. Detects speech segments in
//  16 kHz mono Float32 PCM (CPU inference). Conforms to `VADGating` so it can
//  drive a `StreamingSession`.
//
#if os(iOS)
import Foundation
import ParakeetCore
import CParakeet

public actor FireRedVAD: VADGating {
    /// A detected speech segment (alias of the core `SpeechSegment`).
    public typealias Segment = SpeechSegment

    private let ctx: OpaquePointer

    public init(modelPath: String) throws {
        guard let ctx = firered_vad_init(modelPath) else {
            throw VADError.loadFailed(modelPath)
        }
        self.ctx = ctx
    }

    deinit {
        firered_vad_free(ctx)
    }

    /// Loads the (tiny) VAD model off-main.
    public static func make(modelPath: String) async throws -> FireRedVAD {
        try await Task.detached(priority: .userInitiated) {
            try FireRedVAD(modelPath: modelPath)
        }.value
    }

    /// Detects speech segments. `threshold` = probability threshold,
    /// `minSpeech`/`minSilence` = minimum segment/pause length (s).
    public func detect(_ samples: [Float],
                       threshold: Float = 0.5,
                       minSpeech: Float = 0.2,
                       minSilence: Float = 0.35) -> [SpeechSegment] {
        guard !samples.isEmpty else { return [] }
        return samples.withUnsafeBufferPointer { buf -> [SpeechSegment] in
            var segs: UnsafeMutablePointer<firered_vad_segment>? = nil
            var n: Int32 = 0
            let rc = firered_vad_detect(ctx, buf.baseAddress, Int32(buf.count),
                                        &segs, &n, threshold, minSpeech, minSilence)
            guard let segs else { return [] }
            defer { free(segs) }
            guard rc >= 0, n > 0 else { return [] }
            var out: [SpeechSegment] = []
            out.reserveCapacity(Int(n))
            for i in 0..<Int(n) {
                let s = segs[i]
                out.append(SpeechSegment(start: Double(s.start_sec), end: Double(s.end_sec)))
            }
            return out
        }
    }

    // MARK: - Bundled streaming VAD model

    /// URL of the bundled streaming VAD model (`Bundle.module`). nil if missing.
    public static var bundledModelURL: URL? {
        Bundle.module.url(forResource: "firered-stream-vad", withExtension: "gguf")
    }

    /// Loads the bundled VAD model off-main.
    public static func bundled() async throws -> FireRedVAD {
        guard let url = bundledModelURL else { throw VADError.bundledModelMissing }
        return try await make(modelPath: url.path)
    }
}

#endif
