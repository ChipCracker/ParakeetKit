# ParakeetKit

A reusable Swift Package for **on-device speech recognition and speaker
diarization with parakeet.cpp** (NVIDIA Parakeet TDT + FireRedVAD +
TitaNet/pyannote, via ggml/Metal). Wraps the `Parakeet.xcframework` and
provides clean APIs for:

- **Model download** — an extensible catalog (4 Parakeet TDT 0.6B v3 quants + TitaNet/pyannote diarization models + register your own) and a SwiftUI-free downloader.
- **Live streaming** — a hypothesis ("hyp") preview + committed ("fester") transcript, VAD-gated, surfaced as an `AsyncStream<StreamingEvent>`.
- **Speaker diarization** — live per-segment speaker IDs, word-level speakers + turns in the final pass, persistent named-speaker recognition.
- **Metrics & evaluation** — encoder/decoder run counts + RTF per call and per session, a `WordErrorRate` utility, and a benchmark suite (`scripts/benchmark.sh`).

> **iOS only.** The `Parakeet.xcframework` is a static library with `ios-arm64`
> + `ios-arm64-simulator` slices (no macOS). The pure-logic `ParakeetCore` target
> builds + tests on macOS; the binary-backed `ParakeetKit` builds for iOS.

## Install

```swift
.package(url: "https://github.com/ChipCracker/ParakeetKit.git", from: "1.0.0")
// targets: .product(name: "ParakeetKit", package: "ParakeetKit")
```

## Quick start — offline

```swift
import ParakeetKit

let spec = ParakeetModelCatalog.recommended          // Parakeet TDT 0.6B v3 q4_K
let url = try await ModelDownloader().download(spec) { p in print("download \(Int(p*100))%") }
let engine = try await ParakeetEngine.make(spec: spec, downloadedAt: url)

let samples = try AudioFileLoader.loadSamples(url: wavURL)   // 16 kHz mono
let result = await engine.transcribe(samples)                // ≤ ~20 s clips
// long recordings: NeMo-streamed windows (30 s/5 s), same quality, word timestamps
let long = await engine.transcribeLong(samples)
print(result.text, result.words, result.encoderRuns, result.rtf)
```

## Quick start — live streaming (hyp + committed text)

```swift
let engine = try await ParakeetEngine.make(spec: .recommended, downloadedAt: url)
let live = LiveTranscriber(engine: engine)            // uses the BUNDLED FireRedVAD

for await event in try await live.start() {
    switch event {
    case .hypothesis(let hyp):              // grey, in-flight preview
        print("hyp:", hyp)
    case .committed(_, let full):           // black, finalised ("fester") text
        print("committed:", full)
    case .stats(let s):                     // cumulative encoder/decoder runs
        print("runs:", s.encoderRuns, s.decoderSteps, "rtf:", s.rtf)
    case .speaking(let on): print("speaking:", on)
    case .level(let l):     _ = l           // 0…1 for a level meter / waveform
    case .finalized(let text): print("final:", text)
    case .speaker(let i, let id, let name): // diarization: who said segment i
        print("segment \(i) → speaker \(name ?? "#\(id)")")
    case .finalizedTranscript(let t):       // word timestamps (+ speakers/turns)
        _ = t.speakerTurns
    }
}
// later:
await live.stop()
```

## Speaker diarization (native, on-device)

TitaNet-Large embeddings + online clustering label every committed segment
live (`.speaker` events); the pyannote segmentation final pass adds word-level
speakers and turns to `.finalizedTranscript`. Enrolled voices are recognised
by name across sessions.

```swift
let downloader = ModelDownloader()
let titanet  = try await downloader.download(ParakeetModelCatalog.titanetLarge)      // ~44 MB
let pyannote = try await downloader.download(ParakeetModelCatalog.pyannoteSegmentation) // ~6 MB

let live = LiveTranscriber(
    engine: engine,
    diarization: LiveDiarization(
        titanetModelURL: titanet,
        pyannoteModelURL: pyannote,                       // nil = no final pass
        speakerDBDirectory: mySpeakerProfilesDirectory))  // nil = no name recognition

try await live.enrollSpeaker(name: "christopher", samples: voiceSample) // ≥ ~1 s
```

`Diarizer`, `SpeakerEmbedder`, `PyannoteSegmenter` (ParakeetKit) and
`SpeakerClusterer`, `SpeakerDB`, `PyannotePosteriors` (ParakeetCore, pure
Swift) are public for custom pipelines. TitaNet/pyannote run on CPU —
`DiarizationOptions.threads` (default 2) is the tuning knob; verified
quality numbers live in `benchmarks/README.md`.

`StreamingSession` (in `ParakeetCore`) holds the commit/hypothesis state machine,
is driven by an injected transcriber + `VADGating`, and is fully unit-testable
without the binary. Tunables live in `StreamingConfig` (defaults: 0.8 s endpoint
silence, 16 s max segment, 0.6 s preview step, 0.3 s speech pad, …).

### Streaming cost controls

Previews re-transcribe the buffered segment while you speak; three
`StreamingConfig` knobs keep that cost bounded **without touching the
committed/final text** (verified byte-identical by the pipeline benchmark):

- `previewWindowSeconds` (default 8, `0` = unbounded): previews only see the
  tail window; words that scroll out are frozen into the hypothesis prefix.
  Commits always use the full speech window.
- `previewStepSlowSeconds` / `previewSlowAfterSeconds` (1.2 s / 8 s,
  `.infinity` = off): long segments preview less often.
- `reuseLastPreviewOnCommit` (default on): when the endpoint window equals the
  last preview window, its result is committed without a fresh run (greedy
  decode is deterministic).

### Engine defaults

- `ParakeetEngine.make(useFlashAttention: nil)` enables flash attention
  whenever `useGPU` is on — bit-identical output, ~1.6× faster encoder on
  Metal (upstream-verified). Pass `false`/`true` to override.
- `transcribeLong` uses the NeMo-streamed path (global z-norm, 30 s/5 s
  windows): same WER as the old chunked path at ~27 % lower RTF on long audio.
  `transcribeChunked(chunkSeconds:overlapSeconds:)` keeps the legacy behaviour.

## Custom models

```swift
ParakeetModelCatalog.shared.register(.huggingFace(
    id: "my-asr-q8", displayName: "q8_0", family: "My ASR",
    quantization: .q8_0, repo: "me/My-ASR-GGUF",
    fileName: "my-asr-q8_0.gguf", approxBytes: 700_000_000))
```

## Building / verifying

```bash
bash scripts/build-xcframework.sh        # copy Parakeet.xcframework from parakeet-ios + inject modulemap
                                         # (--rebuild builds upstream first: parakeet + firered-vad + diarization)
swift test                               # pure-logic tests (streaming, diarization core, catalog, WER) on macOS
bash scripts/benchmark.sh                # WER/RTF/cost + diarization benchmarks on the iOS simulator — see benchmarks/README.md
PARAKEET_BENCH_DEST=device bash scripts/benchmark.sh   # same suite on a connected device (Metal; downloads models once)
```

Device/benchmark runs generate `ParakeetBench.xcodeproj` via `xcodegen`
(project.yml; gitignored). Note: while that project exists in the repo root,
xcodebuild prefers it over the SPM package — plain package builds then go
through `-project ParakeetBench.xcodeproj -scheme BenchHost` (or delete the
generated project).

## Publishing a release (remote SPM)

```bash
bash scripts/build-xcframework.sh
bash scripts/package-xcframework.sh parakeet-2   # zips + prints url+checksum
# paste url+checksum into Package.swift, commit, then:
git tag parakeet-2 && git push --tags
gh release create parakeet-2 dist/Parakeet.xcframework.zip
```

The manifest auto-selects: local `Frameworks/Parakeet.xcframework` when present
(or `PARAKEETKIT_LOCAL_XCFRAMEWORK=1`), otherwise the remote `url:`+`checksum:`.

> ⚠️ The diarization backend (titanet/pyannote symbols) exists only in locally
> rebuilt frameworks so far — the published `parakeet-1` binary predates it.
> Remote-SPM consumers need a `parakeet-2` binary release before the
> diarization APIs work for them.

## Notes

- The bundled FireRedVAD model (`firered-stream-vad.gguf`, ~2.2 MB) ships in the package (`Bundle.module`); ASR and diarization models are downloaded at runtime.
- The static archive bundles its own ggml. A standalone ParakeetKit app needs no special linking. An app that uses **both ParakeetKit and LlamaKit** must `-force_load` the parakeet archive (the two ggml copies otherwise collide — static archive vs. LlamaKit's dynamic framework).
- Models: Parakeet TDT 0.6B v3 (CC-BY-4.0), TitaNet-Large (CC-BY-4.0, NVIDIA), pyannote-segmentation-3.0 (MIT) — GGUFs by `cstr`; FireRedVAD (Apache-2.0). Inference: CrispStrobe/CrispASR (parakeet + firered-vad + diarization backends), ggml.
- New in 1.2: `StreamingEvent` gained `.speaker` and `.finalizedTranscript` — exhaustive switches need the two new cases (or a `default`).
