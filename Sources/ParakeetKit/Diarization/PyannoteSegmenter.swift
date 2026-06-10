//
//  PyannoteSegmenter.swift
//  Swift wrapper around the pyannote-segmentation-3.0 C API
//  (CParakeet/pyannote_seg.h): 16 kHz mono PCM → (T, 7) log-softmax
//  posteriors, surfaced as `PyannotePosteriors` (ParakeetCore) for turn
//  extraction and dominant-speaker scoring.
//
#if os(iOS)
import Foundation
import ParakeetCore
import CParakeet

public actor PyannoteSegmenter {
    private let ctx: OpaquePointer

    /// `useGPU` runs front-end, per-layer LSTM input GEMMs and the classifier
    /// head as ggml graphs on Metal; only the sequential LSTM recurrence stays
    /// on CPU (turn-level outputs match the CPU reference).
    public init(modelPath: String, threads: Int = 2, useGPU: Bool = false) throws {
        guard let ctx = pyannote_seg_init_ex(modelPath, Int32(threads), useGPU) else {
            throw ParakeetError.modelLoadFailed(modelPath)
        }
        self.ctx = ctx
    }

    deinit {
        pyannote_seg_free(ctx)
    }

    public static func make(modelPath: String, threads: Int = 2,
                            useGPU: Bool = false) async throws -> PyannoteSegmenter {
        try await Task.detached(priority: .userInitiated) {
            try PyannoteSegmenter(modelPath: modelPath, threads: threads, useGPU: useGPU)
        }.value
    }

    /// Runs segmentation over the buffer. `startTime` shifts the posterior
    /// timeline to absolute session time (seconds).
    public func posteriors(for samples: [Float], startTime: Double = 0) -> PyannotePosteriors? {
        guard !samples.isEmpty else { return nil }
        var frames: Int32 = 0
        guard let buffer = samples.withUnsafeBufferPointer({ buf in
            pyannote_seg_run(ctx, buf.baseAddress, Int32(buf.count), &frames)
        }) else { return nil }
        defer { free(buffer) }
        guard frames > 0 else { return nil }
        let values = Array(UnsafeBufferPointer(start: buffer, count: Int(frames) * 7))
        return PyannotePosteriors(logPosteriors: values, frameCount: Int(frames), startTime: startTime)
    }
}

#endif
