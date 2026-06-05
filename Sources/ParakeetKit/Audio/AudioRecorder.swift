//
//  AudioRecorder.swift
//  Captures the microphone and delivers 16 kHz mono Float32 PCM as a stream
//  (callback per block).
//
#if os(iOS)
import AVFoundation

public final class AudioRecorder {
    private let engine = AVAudioEngine()
    private let targetFormat: AVAudioFormat
    private var converter: AVAudioConverter?
    private var onSamples: (([Float]) -> Void)?
    public private(set) var isRunning = false

    public let targetSampleRate: Double

    public init(targetSampleRate: Double = 16_000) {
        self.targetSampleRate = targetSampleRate
        self.targetFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                          sampleRate: targetSampleRate,
                                          channels: 1,
                                          interleaved: false)!
    }

    /// Requests microphone permission.
    public static func requestPermission() async -> Bool {
        await withCheckedContinuation { cont in
            if #available(iOS 17.0, *) {
                AVAudioApplication.requestRecordPermission { cont.resume(returning: $0) }
            } else {
                AVAudioSession.sharedInstance().requestRecordPermission { cont.resume(returning: $0) }
            }
        }
    }

    /// Starts recording. `onSamples` is called per converted block (on an audio thread).
    public func start(onSamples: @escaping ([Float]) -> Void) throws {
        guard !isRunning else { return }
        self.onSamples = onSamples

        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .measurement, options: [.defaultToSpeaker])
        try session.setActive(true, options: .notifyOthersOnDeactivation)

        let input = engine.inputNode
        let inputFormat = input.outputFormat(forBus: 0)
        converter = AVAudioConverter(from: inputFormat, to: targetFormat)

        input.installTap(onBus: 0, bufferSize: 4096, format: inputFormat) { [weak self] buffer, _ in
            self?.process(buffer)
        }
        engine.prepare()
        try engine.start()
        isRunning = true
    }

    public func stop() {
        guard isRunning else { return }
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
        onSamples = nil
        converter = nil
        isRunning = false
    }

    private func process(_ buffer: AVAudioPCMBuffer) {
        guard let converter, buffer.frameLength > 0 else { return }
        let ratio = targetSampleRate / buffer.format.sampleRate
        let capacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 1
        guard capacity > 0,
              let out = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: capacity) else { return }

        var fed = false
        var error: NSError?
        let status = converter.convert(to: out, error: &error) { _, inStatus in
            if fed {
                inStatus.pointee = .noDataNow
                return nil
            }
            fed = true
            inStatus.pointee = .haveData
            return buffer
        }
        guard status != .error, out.frameLength > 0, let ch = out.floatChannelData else { return }
        let samples = Array(UnsafeBufferPointer(start: ch[0], count: Int(out.frameLength)))
        onSamples?(samples)
    }
}

#endif
