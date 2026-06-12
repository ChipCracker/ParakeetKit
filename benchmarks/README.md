# ParakeetKit Benchmarks

## Diarization auf GPU (Metal)

TitaNet und Pyannote laufen mit `DiarizationOptions.useGPU` (Default: an auf
Geräten, aus im Simulator) als ggml-Graphen auf Metal; bei Pyannote bleibt nur
die sequenzielle LSTM-Rekurrenz auf der CPU — über Gates, die pro Layer in
EINER GEMM für die ganze Sequenz vorprojiziert werden.

Parität + Speedup (M3 Max Metal, C-Paritätstools):

| Modell | CPU-Referenz | Metal-Graph | Speedup | Parität |
|---|---|---|---|---|
| TitaNet (6,5 s Audio) | 9,48 s | **0,030 s** | **312×** | cosine = 1.000000 |
| Pyannote (11 s Audio) | 0,68 s | **0,130 s** | **5,3×** | T identisch, 99,85 % argmax, Turns identisch |

Die Pyannote-Restabweichung (max|Δlogp| 0,34 an 1/650 Übergangs-Frames) ist
F32-Summationsreihenfolge, durch 4 sättigende BiLSTM-Schichten verstärkt —
turn-level wirkungslos. `testGPUParity` (Device-gated) prüft auf dem Gerät
cosine ≥ 0,999 und identische Turn-Sequenzen (±1 Frame). **iPad-Lauf steht
aus** (Gerät war beim letzten Versuch getrennt): `PARAKEET_BENCH_DEST=device
bash scripts/benchmark.sh`.

Simulator-Befund (empirisch verifiziert via `PARAKEET_BENCH_FORCE_GPU=1`):
Der Metal-Graph-Pfad **crasht im Simulator beim Gewichts-Upload**
(`MTLSimDevice newBufferWithLength` → `xpc_shmem_create` → xpc_api_misuse) —
der Sim-Treiber unterstützt ggml-metals Shared-Buffer nicht. Genau deshalb
ist `useGPU` dort per Default aus; der komplette Diarization-CPU-Pfad läuft
im Simulator grün (testLiveAttribution/testFinalPass). Metal-Verifikation:
macOS-Paritätstools (oben) + Gerät.

## Diarization-Testaudio

`Tests/ParakeetKitBenchmarks/Resources/voice-{ryan,serena}.wav` sind zwei
synthetische Sprecher (qwen3-tts CustomVoice, 16 kHz mono, ASR-verifizierte
Transkripte in `BenchEnv`). Regenerieren:

```bash
cd ../parakeet-ios/third_party/CrispASR
cmake -B build-macos -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON \
      -DCRISPASR_BUILD_TESTS=OFF -DCRISPASR_BUILD_SERVER=OFF
cmake --build build-macos --target crispasr-cli -j 8
# Talker: cstr/qwen3-tts-0.6b-customvoice-GGUF (q8_0, 967 MB)
# Codec:  cstr/qwen3-tts-tokenizer-12hz-GGUF (q8_0, 290 MB)
build-macos/bin/crispasr --backend qwen3-tts -m talker.gguf --codec-model codec.gguf \
    --voice ryan --tts "The quick brown fox jumps over the lazy dog near the river bank." \
    --tts-output voice-ryan-24k.wav
afconvert -f WAVE -d LEI16@16000 -c 1 voice-ryan-24k.wav voice-ryan.wav
# Weitere Stimmen: aiden, dylan, eric, ono_anna, ryan, serena, sohee, uncle_fu, vivian
```

`DiarizationBenchmarkTests` prüft mit echten Modellen (TitaNet 44 MB +
pyannote 6 MB, Download via ModelDownloader): Embedding-Konsistenz (gleiche
Stimme → gleiche ID), Sprechertrennung, SpeakerDB-Namensauflösung und den
wortgenauen Final-Pass inkl. Re-Identifikation (ryan → serena → ryan).

Ergebnisse (Sim `…-diarization` / iPad `…-ipad-diar`): identische Qualität
auf beiden — Live-IDs ryan 0/0, serena 1, Name „ryan" aufgelöst; Final-Pass
3 Turns, 31/33 Wörter gelabelt, exakt 2 Sprecher, Re-Identifikation korrekt.
Final-Pass-Zeit auf ~19 s Audio: ~25 s (Sim) / ~30 s (iPad, gedrosselt) —
TitaNet + pyannote laufen CPU-only; `DiarizationOptions.threads` (Default 2)
ist der Tuning-Hebel. Hinweis zur Varianz: Direkt aufeinanderfolgende
Geräteläufe drosseln thermisch (identische Arbeitsmenge, bis 2× Wallclock) —
Zeiten werden deshalb protokolliert, nie geassertet.

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

## Finalize v2 — Diarization-Final-Pass (2026-06-12)

Umbau von `Diarizer.finalize`: overlap-fähige pyannote-Turns (Hysterese je
lokalem Sprecher, gap-merge 0,25 s), purity-maskierte Fenster-Embeddings
(4 s/2 s-Hop ab 6 s Turn-Länge, globales Budget 96), **agglomeratives
Clustering** über alle Fenster (Average-Linkage-Cosine, `maxSpeakers` als
echtes Stop-Kriterium) mit konservativem Turn-Split, ID-Mapping auf die
Session-Cluster (Live-IDs bleiben gültig), Aktivitäts-Tiebreak für
überlappte Wörter + Nearest-Turn-Fallback (≤ 0,5 s) und zentroid-basiertem,
konfliktfreiem Namens-Resolve. Neue Testdaten: `voice-aiden.wav`
(qwen3-tts CustomVoice, 8,5 s).

Simulator iPhone 16 (CPU-Pfad), q4_K + TitaNet + pyannote:

| Test | Metrik | alt | v2 |
|---|---|---|---|
| 3 Sprecher (ryan serena aiden ryan serena) | Wort-Accuracy | 1.000 | 1.000 |
| | Coverage (gelabelte In-Speech-Wörter) | 0.943 | **1.000** |
| | Cluster (Soll 3, beide Wiederkehrer re-identifiziert) | 3 | 3 |
| | finalPass s (inkl. ASR, Sim-CPU) | 41,9 | 52,2 |
| 2 Sprecher (Bestandstest) | labelledWords | 31/33 | **33/33** |
| | distinctSpeakers | 2 | 2 |

Einordnung: Das synthetische Konkat-Audio enthält keine echten
Überlappungen — der Overlap-Teil (per-Sprecher-Turns, Purity-Masking)
zeigt seinen Gewinn erst auf realen Gesprächen; die Tabelle belegt
Regressionsfreiheit plus den Coverage-Gewinn des Nearest-Fallbacks.
Mehrkosten ~10 s im Sim-CPU-Pfad durch zusätzliche Fenster-Embeddings;
mit Metal-TitaNet auf Geräten (~0,03–0,1 s/Embedding) unerheblich.
Neue Core-Units decken AHC (Gruppen, maxClusters-Zwang, degenerierte
Vektoren), perSpeakerTurns (Overlap, gap-merge), pureRange und
speakerActivity ab.

### Overlap-Test (Nachtrag, 2026-06-12)

`testFinalPassOverlappingSpeech`: zwei neue qwen3-tts-Stimmen
(`voice-dylan.wav` 13,6 s, `voice-sohee.wav` 10,2 s) werden mit **3 s
echtem Doppelsprechen** gemischt (Ränder still-getrimmt). Ergebnis
(Sim-CPU): `maxTurnOverlapSeconds 3.26` — der v2-Pass bildet die
Überlappung als zeitlich überlappende Turns zweier Sprecher ab (der
alte argmax-Pass konnte das strukturell nicht) — bei `soloAccuracy
1.000` und exakt 2 Clustern auf den Solo-Strecken. Der Lauf deckte
zudem einen v2-Bug auf: pyannote trackt dieselbe Stimme zeitweise auf
zwei lokalen Slots (identische Aktivitätsspuren), was nach dem
ID-Mapping Duplikat-Turns erzeugte — gleiche-Sprecher-Turns werden
jetzt gemergt (Turns: 3-Sprecher 10→8, 2-Sprecher 6→5, Metriken
unverändert).

### Long-Audio-Fix (2026-06-12)

Befund: Bei langen Offline-Aufnahmen (~17 min) wurde die App während
„Sprecher zuordnen …“ vom System beendet (Jetsam). Ursache: pyannote
lief über das GESAMTE Audio in einem Stück — die Aktivierungen wachsen
linear mit T (SincNet-Zwischenausgabe allein ~0,5 GB bei 17 min, der
ggml-Graph hält mehrere solcher Tensoren gleichzeitig).

Fix: Die Segmentierung läuft jetzt in **10-s-Fenstern** (zugleich die
Trainingsdomäne von pyannote-3.0); lokale Slots gelten pro Fenster,
die globale Identität kommt wie gehabt aus TitaNet+AHC, und der
Same-Speaker-Merge (Toleranz 0,3 s) heilt die Fensternähte.

Neuer Wächter `testFinalPassLongAudio` (3,9 min Wechselrede, Sim-CPU):
accuracy **1.000**, 2 Cluster, 40 Turns; ASR 26,7 s, Finalize 383 s —
letzteres ist der CPU-Referenzpfad des Simulators (TitaNet-Hand-Loop);
auf Geräten rechnet Metal-TitaNet dieselben ~50 Embeddings in Sekunden.
Bestandstests unverändert grün (3-Sprecher 1.000/1.000, 2-Sprecher
33/33). Bekannter Tradeoff: Fenstergrenzen können die ERKANNTE
Overlap-Spanne verkürzen (Overlap-Test: ~3 s konstruiert → 0,56 s
erkannt, wenn die Zone eine 10-s-Grenze überspannt) — der Test prüft
seither die Existenz überlappender Turns (≥ 0,4 s); volle Spannen
bräuchten überlappende Segmentierungsfenster (möglicher Folgeschritt).
