# ParakeetKit Benchmarks

Verifiziert, dass Performance-Optimierungen die Erkennungsqualität nicht
verschlechtern. Zwei Ebenen:

1. **Pipeline-Benchmark** (macOS, `swift test`, deterministisch, ohne Binary) —
   Kostenprofil der `StreamingSession`: Wie viele transcribe-Aufrufe und
   transkribierte Audio-Sekunden erzeugt ein Sprech-Szenario? Misst die
   Redundanz der Preview-/Commit-Pipeline; `committedText` muss über
   Optimierungen hinweg identisch bleiben.
2. **Engine-Benchmarks** (iOS-Simulator, echte Inferenz, CPU-Pfad) —
   WER gegen den bekannten jfk.wav-Referenztext (11 s; Long-Audio = 6×
   konkateniert, E2E = 3× mit 1,2-s-Pausen), RTF, encoderRuns/decoderSteps.
   Hinweis: Im Simulator läuft ggml ohne Metal — Zeiten sind CPU-Zahlen
   (Trend-Indikator); Qualitäts-Asserts (WER, Text-Parität) sind die harten
   Kriterien. Metal-Speedups (z. B. Flash Attention 1,61×, upstream
   bit-identisch verifiziert) zeigen sich nur auf echten Geräten.

## Ausführen

```bash
bash scripts/benchmark.sh
# PARAKEET_BENCH_MODEL=…/mein-modell.gguf  PARAKEET_BENCH_SIM="iPhone 17 Pro"  PARAKEET_BENCH_LABEL=mein-label
```

Default-Modell: `../parakeet-ios/vendor/models/parakeet-tdt-0.6b-v3-q4_k.gguf`.
Ergebnisse landen als JSON unter `benchmarks/results/<timestamp>-<label>/`:
`pipeline-benchmark.json`, `single-shot.json`, `long-audio.json`,
`e2e-streaming.json`.

## Ergebnisse

Simulator = iPhone 17 Pro (CPU-Pfad, Host: Apple Silicon), Modell q4_K.

### Baseline (`20260610-145811-baseline`, vor Optimierungen)

| Benchmark | Kennzahlen |
|---|---|
| single-shot (jfk 11 s) | WER 0 · RTF 0,109 · 1 encoderRun |
| long-audio (jfk×6, 66 s, `transcribeLong` = chunked 20 s/2 s) | WER 0 · RTF 0,142 · 4 encoderRuns |
| e2e-streaming (jfk×3 + 1,2 s Gaps, 33,7 s) | committed/finalized WER 0 · **63 transcribe-Aufrufe** · **223,4 s transkribiert** · **27,2 s Gesamt-Inferenz** (Commits allein: 3,66 s) |
| pipeline (synthetisch, 4 s/8 s/16 s Utterances, 31,2 s) | **54 Aufrufe** (49 Previews) · **311,5 s Preview-Audio** + 30,5 s Commit-Audio |

Interpretation: Die Live-Pipeline transkribiert das ~6,6-fache des ingestierten
Audios (synthetisch bis ~11×, da die 16-s-Utterance den quadratischen
Preview-Aufwand zeigt). Genau hier setzen die Optimierungen B3–B5 an; B1/B2
verbessern die Engine-Seite (Flash Attention, streamed-API).

### B1 Flash Attention (`…-opt1-flash`)

Transkript-Parität bestätigt (normalisiert identisch, WER 0). CPU-Zeiten im
Simulator streuen stark (Einzelläufe ±20–30 % in beide Richtungen); der Test
misst daher den Median aus 3 Läufen: flash 1,15 s vs. no-flash 1,86 s — auf CPU
also eher leicht vorteilhaft (Upstream: ~+10 % CPU), auf Metal klar belegt
(1,61×, bit-identisch, upstream PERFORMANCE.md). `useFlashAttention: nil`
(Default) koppelt Flash an `useGPU`: an auf Geräten (Metal), aus auf dem
CPU-/Simulator-Pfad — konservativ, explizit überschreibbar.
