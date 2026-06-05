//
//  ModelDownloader.swift
//  Framework-agnostic GGUF downloader (no SwiftUI). Progress via an
//  `AsyncThrowingStream` or a one-shot async call. SwiftUI consumers can wrap
//  this in their own `ObservableObject`.
//
import Foundation

public struct DownloadConfig: Sendable {
    /// Where models are stored. Default: Application Support/Models/Parakeet/.
    public var directory: URL
    public var waitsForConnectivity: Bool

    public init(directory: URL = ModelDownloader.defaultDirectory,
                waitsForConnectivity: Bool = true) {
        self.directory = directory
        self.waitsForConnectivity = waitsForConnectivity
    }
}

public enum DownloadEvent: Sendable {
    case progress(Double)   // 0...1
    case finished(URL)
}

public enum DownloadState: Sendable, Equatable {
    case notDownloaded
    case downloading(Double)
    case ready(URL)
    case failed(String)
}

public final class ModelDownloader: Sendable {
    /// Application Support/Models/Parakeet/, auto-created.
    public static let defaultDirectory: URL = {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Models", isDirectory: true)
            .appendingPathComponent("Parakeet", isDirectory: true)
        try? FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        return base
    }()

    public let directory: URL
    public let waitsForConnectivity: Bool

    public init(config: DownloadConfig = .init()) {
        self.directory = config.directory
        self.waitsForConnectivity = config.waitsForConnectivity
        try? FileManager.default.createDirectory(at: config.directory, withIntermediateDirectories: true)
    }

    // MARK: Queries

    public func localURL(for spec: ParakeetModelSpec) -> URL {
        directory.appendingPathComponent(spec.fileName)
    }

    public func isDownloaded(_ spec: ParakeetModelSpec) -> Bool {
        FileManager.default.fileExists(atPath: localURL(for: spec).path)
    }

    public func state(for spec: ParakeetModelSpec) -> DownloadState {
        isDownloaded(spec) ? .ready(localURL(for: spec)) : .notDownloaded
    }

    @discardableResult
    public func delete(_ spec: ParakeetModelSpec) throws -> Bool {
        let url = localURL(for: spec)
        guard FileManager.default.fileExists(atPath: url.path) else { return false }
        try FileManager.default.removeItem(at: url)
        return true
    }

    // MARK: Download

    /// Streaming download. Cancel by terminating the stream. Atomic move into place.
    public func download(_ spec: ParakeetModelSpec) -> AsyncThrowingStream<DownloadEvent, Error> {
        let dest = localURL(for: spec)
        let waits = waitsForConnectivity
        let remote = spec.downloadURL
        return AsyncThrowingStream { continuation in
            let delegate = DownloadDelegate(destination: dest, continuation: continuation)
            let config = URLSessionConfiguration.default
            config.waitsForConnectivity = waits
            let session = URLSession(configuration: config, delegate: delegate, delegateQueue: nil)
            let task = session.downloadTask(with: remote)
            continuation.onTermination = { _ in
                task.cancel()
                session.invalidateAndCancel()
            }
            task.resume()
        }
    }

    /// One-shot convenience: awaits the final URL, reporting progress via callback.
    @discardableResult
    public func download(_ spec: ParakeetModelSpec,
                         onProgress: (@Sendable (Double) -> Void)? = nil) async throws -> URL {
        for try await event in download(spec) {
            switch event {
            case .progress(let p): onProgress?(p)
            case .finished(let url): return url
            }
        }
        throw ParakeetError.downloadFailed("download ended without a file")
    }
}

// MARK: - URLSession delegate bridge

private final class DownloadDelegate: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private let destination: URL
    private let continuation: AsyncThrowingStream<DownloadEvent, Error>.Continuation

    init(destination: URL, continuation: AsyncThrowingStream<DownloadEvent, Error>.Continuation) {
        self.destination = destination
        self.continuation = continuation
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                    didWriteData bytesWritten: Int64, totalBytesWritten: Int64,
                    totalBytesExpectedToWrite: Int64) {
        guard totalBytesExpectedToWrite > 0 else { return }
        continuation.yield(.progress(Double(totalBytesWritten) / Double(totalBytesExpectedToWrite)))
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                    didFinishDownloadingTo location: URL) {
        if let http = downloadTask.response as? HTTPURLResponse, !(200...299).contains(http.statusCode) {
            continuation.finish(throwing: ParakeetError.downloadFailed("HTTP \(http.statusCode)"))
            return
        }
        do {
            try? FileManager.default.removeItem(at: destination)
            try FileManager.default.moveItem(at: location, to: destination)
            continuation.yield(.finished(destination))
            continuation.finish()
        } catch {
            continuation.finish(throwing: ParakeetError.downloadFailed(error.localizedDescription))
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask,
                    didCompleteWithError error: Error?) {
        guard let error else { return }
        if (error as NSError).code == NSURLErrorCancelled { continuation.finish(); return }
        continuation.finish(throwing: ParakeetError.downloadFailed(error.localizedDescription))
    }
}
