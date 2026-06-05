//
//  AudioFileLoader.swift
//  Loads an audio file (WAV/M4A/MP3/CAF …) and converts it to 16 kHz mono
//  Float32 — for offline import and the bundled sample.
//
#if os(iOS)
import AVFoundation
import ParakeetCore

public enum AudioFileLoader {
    /// Reads the whole file and returns 16 kHz mono Float32 samples.
    public static func loadSamples(url: URL, targetSampleRate: Double = 16_000) throws -> [Float] {
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }

        let file: AVAudioFile
        do {
            file = try AVAudioFile(forReading: url)
        } catch {
            throw AudioLoadError.cannotOpen(url.lastPathComponent)
        }

        let inputFormat = file.processingFormat
        guard let targetFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                               sampleRate: targetSampleRate,
                                               channels: 1,
                                               interleaved: false),
              let converter = AVAudioConverter(from: inputFormat, to: targetFormat) else {
            throw AudioLoadError.conversionFailed
        }

        let frameCount = AVAudioFrameCount(file.length)
        guard frameCount > 0,
              let inputBuffer = AVAudioPCMBuffer(pcmFormat: inputFormat, frameCapacity: frameCount) else {
            return []
        }
        try file.read(into: inputBuffer)

        let ratio = targetSampleRate / inputFormat.sampleRate
        let outCapacity = AVAudioFrameCount(Double(inputBuffer.frameLength) * ratio) + 4096
        guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: outCapacity) else {
            throw AudioLoadError.conversionFailed
        }

        var fed = false
        var error: NSError?
        let status = converter.convert(to: outputBuffer, error: &error) { _, inStatus in
            if fed {
                inStatus.pointee = .endOfStream
                return nil
            }
            fed = true
            inStatus.pointee = .haveData
            return inputBuffer
        }
        guard status != .error, let ch = outputBuffer.floatChannelData else {
            throw AudioLoadError.conversionFailed
        }
        return Array(UnsafeBufferPointer(start: ch[0], count: Int(outputBuffer.frameLength)))
    }
}

#endif
