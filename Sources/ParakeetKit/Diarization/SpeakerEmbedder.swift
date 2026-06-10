//
//  SpeakerEmbedder.swift
//  Swift wrapper around the TitaNet-Large C API (CParakeet/titanet.h):
//  16 kHz mono PCM → 192-d L2-normalized speaker embedding. CPU inference;
//  an 11 s utterance embeds in well under a second on modern devices.
//
#if os(iOS)
import Foundation
import ParakeetCore
import CParakeet

public actor SpeakerEmbedder {
    public static let dimension = 192
    /// TitaNet needs a minimum of context to produce a stable embedding.
    public static let minimumSeconds: Double = 0.5

    private let ctx: OpaquePointer

    public init(modelPath: String, threads: Int = 2) throws {
        guard let ctx = titanet_init(modelPath, Int32(threads)) else {
            throw ParakeetError.modelLoadFailed(modelPath)
        }
        self.ctx = ctx
    }

    deinit {
        titanet_free(ctx)
    }

    /// Loads the (small) model off-main.
    public static func make(modelPath: String, threads: Int = 2) async throws -> SpeakerEmbedder {
        try await Task.detached(priority: .userInitiated) {
            try SpeakerEmbedder(modelPath: modelPath, threads: threads)
        }.value
    }

    /// Extracts the speaker embedding, or nil for too-short/failed windows.
    public func embed(_ samples: [Float]) -> [Float]? {
        guard Double(samples.count) / 16_000 >= Self.minimumSeconds else { return nil }
        var out = [Float](repeating: 0, count: Self.dimension)
        let rc = samples.withUnsafeBufferPointer { buf in
            titanet_embed(ctx, buf.baseAddress, Int32(buf.count), &out)
        }
        return rc > 0 ? out : nil
    }
}

#endif
