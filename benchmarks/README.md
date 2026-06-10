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

### B2 `transcribeLong` → streamed 30 s/5 s (`…-opt2-streamed`)

jfk×6 (66 s, lückenlos):

| Variante | WER | RTF | enc | dec |
|---|---|---|---|---|
| chunked 20/2 (alter Default) | 0 | 0,160 | 4 | 548 |
| **streamed 30/5 (neuer Default)** | **0** | **0,117** | 3 | 269 |
| streamed Binary-Heuristik 30/2 | 0,167 | 0,106 | 3 | 260 |

Befund: Die streamed-API ist bei gleicher Qualität ~27 % schneller als chunked
(globale z-norm, ein Mel-Pass, kein doppelt dekodierter Overlap — dec halbiert).
**Aber:** Die im Binary eingebaute Default-Heuristik (30 s Chunk, 2 s Overlap)
verliert nachweislich Wörter an den Chunk-Grenzen (WER 0,167) — deshalb setzt
`transcribeLong` explizit 30 s/5 s statt der Heuristik. Die Heuristik-Variante
läuft als Watchdog im Benchmark mit: Fällt ihr WER nach einem
xcframework-Update auf ~0, kann wieder delegiert werden.

### Endstand: Baseline → alle Optimierungen (`…-optimized`)

Qualität in allen Benchmarks unverändert: WER 0 (single-shot, long-audio,
E2E committed UND finalized), committedText im Pipeline-A/B byte-identisch.

| Metrik | Baseline | Optimiert | Δ |
|---|---|---|---|
| E2E (jfk×3, 11-s-Utterances): transcribe-Aufrufe | 63 | 56 | −11 % |
| E2E: transkribierte Sekunden | 223,4 | 198,8 | −11 % |
| E2E: Gesamt-Inferenzzeit (Sim-CPU) | 27,2 s | 25,5 s | −6 % |
| Pipeline (4/8/16-s-Utterances): Aufrufe | 54 | 43 | −20 % |
| Pipeline: transkribierte Sekunden | 342 | 213 | −38 % |
| transcribeLong 66 s: RTF | 0,160 (chunked) | 0,116 (streamed) | −27 % |

Einordnung: Der Pipeline-Gewinn skaliert mit der Utterance-Länge — Deckel und
Kadenz greifen ab `previewWindowSeconds`/`previewSlowAfterSeconds` (je 8 s).
Die 11-s-jfk-Utterances im E2E profitieren nur im letzten Drittel (−11 %); die
16-s-Utterance im synthetischen Szenario zeigt −38 %. Wer mehr Einsparung will,
senkt `previewWindowSeconds`/`previewSlowAfterSeconds` (Kosten: kürzerer
Preview-Kontext bzw. trägere Hyp-Updates — der committed Text bleibt davon
unberührt).

### Physisches Gerät: iPad Air 13″ M3, Metal (`…-ipad-m3`)

`PARAKEET_BENCH_DEST=device bash scripts/benchmark.sh` — Host-App via
xcodegen/project.yml (SPM-Test-Bundles laufen auf Geräten nicht tool-hosted),
Modell wird beim ersten Lauf aufs Gerät geladen (~466 MB, gecacht).

| Benchmark | iPad M3 (Metal) | Simulator (CPU) |
|---|---|---|
| single-shot: RTF | **0,054** (WER 0) | 0,108 |
| long-audio chunked-20/2 | WER **0,106** · RTF 0,075 | WER 0 · RTF 0,160 |
| long-audio **streamed-30/5 (Default)** | **WER 0 · RTF 0,050** | WER 0 · RTF 0,116 |
| E2E: Gesamt-Inferenz (33,7 s Audio) | **14,6 s** (WER 0/0) | 25,5 s |
| flash-parity (Median of 3) | 0,56 s ≈ 0,55 s (neutral) | 1,15 s vs. 1,86 s |

Zwei Gerätebefunde:
1. **Der alte chunked-Pfad driftet auf Metal real** (WER 0,106 auf jfk×6 — im
   CPU-Sim noch 0): Die per-chunk z-norm ist numerisch fragil. Der neue
   streamed-30/5-Default ist auf dem Gerät gleichzeitig fehlerfrei UND 33 %
   schneller — B2 ist dort eine echte Qualitätsverbesserung.
2. **Flash Attention ist auf M3 + q4_K zeitneutral** (Parität bestätigt). Der
   upstream-1,61× wurde auf M1 mit F16 gemessen, wo der Attention-Anteil
   dominiert; bei q4_K dominieren Quant-Matmuls/im2col. Default (an bei GPU)
   bleibt — qualitätsidentisch.

### F16 auf dem iPad M3 (`…-ipad-m3-f16`)

`PARAKEET_BENCH_MODEL_ID=f16` lädt das 1,26-GB-F16-GGUF aufs Gerät. Vergleich
(beide Metal, gleiche Audio-Eingaben):

| Metrik | F16 | q4_K |
|---|---|---|
| single-shot: RTF | **0,050** (WER 0) | 0,054 (WER 0) |
| long-audio streamed-30/5: RTF | **0,049** (WER 0) | 0,050 (WER 0) |
| E2E: Gesamt-Inferenz (33,7 s) | 13,9 s | 14,6 s |
| flash-parity (Median of 3) | 0,555 s ≈ 0,553 s | 0,560 s ≈ 0,550 s |
| Watchdog Binary-Heuristik | WER 0,235 | WER 0,167 |

Befunde:
- **F16 ist nur ~5 % schneller als q4_K** bei identischer Qualität (WER 0) —
  und 2,7× größer (1,26 GB vs. 467 MB). **q4_K bleibt die richtige
  Default-Empfehlung** für on-device.
- **Flash Attention ist auch mit F16 auf dem M3 zeitneutral.** Der
  upstream-1,61× (M1) reproduziert sich auf der M3-GPU-Generation generell
  nicht — der Gewinn dort stammte primär aus Kernel-Launch-Overhead, der auf
  neueren GPUs/Metal-Runtimes deutlich kleiner ist. Output bleibt in allen
  Konfigurationen identisch; der GPU-gekoppelte Default ist damit weiterhin
  unbedenklich.
- Test-Infrastruktur: 5 sequenzielle 1,26-GB-Engine-Loads sprengen das
  iPadOS-Prozesslimit (`posix_memalign failed`) — Single-Shot und Long-Audio
  teilen sich deshalb eine Engine-Instanz pro Suite.
