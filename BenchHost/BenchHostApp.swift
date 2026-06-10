//
//  BenchHostApp.swift
//  Empty host application for on-device benchmark runs. Physical devices
//  cannot run SPM test bundles "tool-hosted" — XCTest needs a host app there.
//  Generated into ParakeetBench.xcodeproj via `xcodegen` (see project.yml);
//  scripts/benchmark.sh drives it with PARAKEET_BENCH_DEST=device.
//
import SwiftUI

@main
struct BenchHostApp: App {
    var body: some Scene {
        WindowGroup {
            Text("ParakeetKit Benchmark Host")
                .font(.headline)
                .padding()
        }
    }
}
