# CREMA-ASIS

This repository contains the code for our paper:

***When Vocal Tone and Literal Meaning Diverge: An Acoustic–Semantic Incongruity Study for Large Audio–Language Models***

Accepted to the Findings of EMNLP 2026.


<br>

The project investigates whether large audio-language models (LALMs) can detect incongruity between a speaker's emotional tone and the literal meaning of what they say.

We introduce CREMA-ASIS, a dataset of ~77k synthetic utterances where acoustic emotion and semantic sentiment are independently controlled, covering both incongruous (e.g., angry voice saying something positive) and congruous conditions. 

Using this dataset, we evaluate three models: (1) Qwen2-Audio, (2) Audio-Flamingo3, and (3) Kimi-Audio, both off-the-shelf and after LoRA fine-tuning. We also perform linear probing of internal representations to understand how these models process acoustic vs. semantic information.

---
## Dataset

The download link for CREMA-ASIS will be provided upon acceptance.

## Project structure

```
CREMA-ASIS/
├── configs/
│   ├── data/                 # LISTEN label mapping
│   ├── models/               # per-model YAML configs
│   ├── probing/              # probing hyperparameters
│   ├── prompts/              # prompt templates
│   └── training/             # fine-tuning configs
├── data/                     # put your data here (see below)
├── scripts/                  # entry points for each pipeline stage
├── src/                      # library code
├── third_party/              # vendored IndexTTS2 + Kimi-Audio code (see its README)
├── weights_used_for_study/   # LoRA checkpoints used for the paper
├── finetuned_models/         # LoRA checkpoints (written by finetune.py)
├── embedding_cache/          # layer embeddings (written by extract_embeddings.py)
├── probe_results/            # probing outputs (written by run_probing.py)
└── results/                  # evaluation outputs (written by evaluate.py)
```

| Script | Stage |
| --- | --- |
| `filter_sentences.py` | 0. GPT-4o sentence selection |
| `generate_data.py` / `generate_data_from_csv.py` | 1. TTS generation |
| `filter_data.py` | 2. AER filtering + retention policy |
| `compute_wer.py` | 2. Transcription check / WER |
| `finetune.py` | 3. LoRA fine-tuning |
| `evaluate.py` | 4. Inference |
| `compute_metrics.py` | 4b. Scoring |
| `prepare_listen.py` | 4c. Out-of-domain data prep |
| `extract_embeddings.py` | 5. Layer-wise embeddings |
| `run_probing.py` | 6. Linear probing |

---

## Setup
First, install PyTorch for your CUDA version from https://pytorch.org/get-started/locally/, then install the base dependencies:
```bash
pip install -r requirements.txt
```

The three models have conflicting `transformers` requirements, so the version you need depends on which model you want to run. The `requirements.txt` pins `transformers==4.46.1` (Qwen2-Audio). Override it for the other models as follows.


**Qwen2-Audio**
```bash
pip install transformers==4.46.1
```

**Audio-Flamingo3**
```bash
pip install --upgrade "git+https://github.com/huggingface/transformers" accelerate
```

**Kimi-Audio**

Follow the official setup instructions at https://github.com/MoonshotAI/Kimi-Audio/tree/master?tab=readme-ov-file#getting-started

For Kimi-Audio multi-GPU fine-tuning, also install DeepSpeed:
```bash
pip install deepspeed
```


**IndexTTS2** (only needed for data generation)

Follow the setup at https://github.com/index-tts/index-tts

---

## Data

The pipeline expects data under a `data/` directory in the project root. The key files you need are:

```
data/
├── crema-asis/
│   └── cremad-sync-wsad/     # CREMA-ASIS generated audio (WAV)
├── CREMA-D/
│   └── AudioWAV_en/          # original CREMA-D audio (TTS speaker references)
├── CREMA-ASIS_meta.csv       # full CREMA-ASIS manifest, all splits
├── CREMA-ASIS_test.csv       # test split, WER-filtered (evaluation)
├── CREMA-ASIS_sentences.csv  # unique sentences + GPT-4o filter rationale
└── crema-d_en_split.csv      # original CREMA-D split (fine-tuning)
```

`CREMA-ASIS_meta.csv` is the single source of truth: one row per generated
utterance, with the reference clip, the actor, the acoustic emotion and
semantic sentiment labels, the split, the AER prediction that let the sample
through filtering, and the Whisper transcript with its WER.
`CREMA-ASIS_test.csv` is the evaluation set: the `split == "test"` rows of that
file with `wer < 0.5` already applied, 5,879 of 6,171. Evaluate against it
directly — no further filtering is needed.

The manifest keeps every row, including the high-WER ones, so the full dataset
counts stay reproducible. `scripts/evaluate.py` applies the same `wer < 0.5` cut
by default, which is a no-op on the pre-filtered test file but does the right
thing if you point it at the manifest instead. Pass `--no-wer-filter` to disable
it.

If you also want to fine-tune on MELD, place it at `data/MELD.Raw/` following the expected directory layout (`meld_train.csv`, `meld_val.csv`, `train_wav/`, `dev_wav/`).

---

## Pipeline

The pipeline runs in roughly this order:

0. Filter candidate sentences with GPT-4o (before generation)
1. Generate synthetic audio
2. Filter by acoustic quality (and check TTS transcription accuracy)
3. Fine-tune
4. Evaluate (and score the predictions)
5. Extract embeddings
6. Run probing

---

### 0. Sentence selection

Sentences come from GoEmotions and are screened with GPT-4o before use: a
sentence is dropped if it does not carry the target sentiment, is offensive, or
only exists in written form. The prompt is in
`configs/prompts/sentence_selection.txt`.

Needs an OpenAI API key:
```bash
export OPENAI_API_KEY=...

python scripts/filter_sentences.py \
    --csv data/goemotions_candidates.csv \
    --sentence-column sentence \
    --sentiment-column sentiment \
    --output data/goemotions_filtered.csv
```

The sentences that survived this stage, with the model's stated reason, are
released in `data/CREMA-ASIS_sentences.csv`.

---

### 1. Data generation

The CREMA-ASIS dataset is built using IndexTTS2 TTS with voice cloning. The idea is to generate speech where we independently control the acoustic emotion (via a CREMA-D reference clip that carries the target emotion) and the semantic content (via a sentence from GoEmotions). This lets us create utterances where the two are incongruous — e.g., an angry-sounding voice saying something positive — at a scale that can't be obtained from natural data. After generation, an acoustic emotion recognition (AER) model is used to filter out samples where the intended emotion clearly wasn't conveyed (see step 2).

For a single sample:
```bash
python scripts/generate_data.py \
    --cfg checkpoints/config.yaml \
    --model-dir checkpoints \
    --speaker data/CREMA-D/AudioWAV_en/1001_DFA_ANG_XX.wav \
    --text "I appreciate it, that's good to know." \
    --output data/samples/output.wav \
    --emotion 0 0 0 0 0 0 0 0
```

The `--speaker` clip is the **original CREMA-D** recording that carries the
target acoustic emotion; the emotion vector stays all-zero so the tone comes
from the reference voice alone.

For batch generation from a CSV manifest (recommended):

The CSV should have columns: `audio_path`, `output_text`, `emo_vector` (8-dim list), `output_name`.

```bash
python scripts/generate_data_from_csv.py \
    --cfg checkpoints/config.yaml \
    --model-dir checkpoints \
    --speaker-dir data/CREMA-D/AudioWAV_en \
    --csv data/CREMA-ASIS_meta.csv \
    --output-dir data/crema-asis/cremad-sync-wsad \
    --skip-existing
```

---

### 2. Data filtering

After generation, run an acoustic emotion recognition (AER) model over the output to identify samples where the intended emotion clearly wasn't reproduced. The AER uses a Whisper-large-v3-based classifier by default.

```bash
python scripts/filter_data.py \
    --wav-dir data/crema-asis/cremad-sync-wsad \
    --output-csv data/acoustic_detection.csv \
    --kept-csv data/acoustic_detection_kept.csv
```

This writes the AER prediction for every file, plus the intended emotion
(parsed from the filename) and a `keep` flag.

The AER is imperfect, so a sample is kept when its prediction is either the
target emotion or a perceptually adjacent one; only clear mismatches are
dropped. The accepted pairs are in `RETENTION_POLICY` in
`src/data/filtering.py`:

| Target | Accepted AER predictions |
| --- | --- |
| happy | happy, surprised, neutral |
| neutral | neutral, angry, happy |
| sad | neutral, sad |
| disgust | surprised, neutral |
| angry | angry, surprised, neutral |

Pass `--no-policy` to write raw predictions without applying the rule.

#### Transcription check

The generated speech is also transcribed with `whisper-base.en` and compared
against the sentence that was requested, which is where the `wer` column in the
manifest comes from:

```bash
python scripts/compute_wer.py \
    --csv data/CREMA-ASIS_meta.csv \
    --data-dir data/crema-asis/cremad-sync-wsad \
    --output data/CREMA-ASIS_meta_wer.csv
```

The same script scores a LALM's own transcripts with `--score-only`, which is
how the transcription results in the paper were produced.

---

### 3. Fine-tuning

LoRA fine-tuning is supported for all three models. Checkpoints are saved under `finetuned_models/`.

**Qwen2-Audio**
```bash
python scripts/finetune.py \
    --model qwen2-audio \
    --model-config configs/models/qwen2_audio.yaml \
    --train-config configs/training/default.yaml \
    --output-dir finetuned_models/qwen2-audio-lora \
    --datasets cremad_annotated cremad_base meld
```

**Audio-Flamingo3**
```bash
python scripts/finetune.py \
    --model audio-flamingo3 \
    --model-config configs/models/audio_flamingo3.yaml \
    --train-config configs/training/default.yaml \
    --output-dir finetuned_models/audio-flamingo3-lora \
    --datasets cremad_annotated cremad_base meld
```

**Kimi-Audio**

Kimi-Audio requires DeepSpeed for multi-GPU fine-tuning:
```bash
deepspeed --num_gpus=1 scripts/finetune.py \
    --model kimi-audio \
    --model-config configs/models/kimi_audio.yaml \
    --train-config configs/training/training_kimi_audio.yaml \
    --output-dir finetuned_models/kimi-audio-lora \
    --datasets cremad_annotated cremad_base meld
```

---

### 4. Evaluation

Evaluate a model on the test set. Run without `--lora-path` for the base model, or with it for a fine-tuned checkpoint.

`evaluate.py` writes one row per sample with the predicted `acoustic_emotion`
and `semantic_sentiment`; it does not score them. Use `scripts/compute_metrics.py`
for that (below). Rows with `wer >= 0.5` are dropped by default, matching the
paper; pass `--no-wer-filter` to keep them.

**Qwen2-Audio (base)**
```bash
python scripts/evaluate.py \
    --model qwen2-audio \
    --model-config configs/models/qwen2_audio.yaml \
    --data data/CREMA-ASIS_test.csv \
    --data-dir data/crema-asis/cremad-sync-wsad \
    --output results/qwen2_base.csv
```

**Qwen2-Audio (fine-tuned)**
```bash
python scripts/evaluate.py \
    --model qwen2-audio \
    --model-config configs/models/qwen2_audio.yaml \
    --lora-path finetuned_models/qwen2-audio-lora/checkpoint \
    --data data/CREMA-ASIS_test.csv \
    --data-dir data/crema-asis/cremad-sync-wsad \
    --output results/qwen2_lora.csv
```

**Audio-Flamingo3 (base)**
```bash
python scripts/evaluate.py \
    --model audio-flamingo3 \
    --model-config configs/models/audio_flamingo3.yaml \
    --data data/CREMA-ASIS_test.csv \
    --data-dir data/crema-asis/cremad-sync-wsad \
    --output results/audio_flamingo3_base.csv
```

**Kimi-Audio (base)**
```bash
python scripts/evaluate.py \
    --model kimi-audio \
    --model-config configs/models/kimi_audio.yaml \
    --data data/CREMA-ASIS_test.csv \
    --data-dir data/crema-asis/cremad-sync-wsad \
    --output results/kimi_audio_base.csv
```

---

### 4b. Scoring the predictions

`compute_metrics.py` turns prediction CSVs into the reported numbers:
acoustic accuracy, semantic accuracy, and dual-condition accuracy (both correct
for the same sample). Samples the model failed to process have empty
predictions and are counted as incorrect.

```bash
python scripts/compute_metrics.py \
    --predictions results/qwen2_base.csv results/qwen2_lora.csv
```

Add `--per-condition` for precision / recall / F1 broken down by
acoustic–semantic pair, grouped into incongruous, congruous, and
neutral-associated conditions:

```bash
python scripts/compute_metrics.py \
    --predictions results/qwen2_base.csv \
    --per-condition \
    --output results/metrics_summary.csv
```

For single-label runs (the joint emotion/sentiment setting on MELD, or the
out-of-domain LISTEN subsets):

```bash
python scripts/compute_metrics.py \
    --predictions results/qwen2_lora_meld.csv \
    --single-task --true-column Emotion --pred-column acoustic_emotion
```

---

### 4c. Out-of-domain evaluation (LISTEN)

Download the LISTEN_full test split from
[VibeCheck1/LISTEN_full](https://huggingface.co/datasets/VibeCheck1/LISTEN_full)
(`data/test-00000-of-00001.parquet`).

LISTEN_full is long-format: one row per (sample, question), with columns
`id`, `question`, and `answer`. Which modality a row belongs to is encoded in
the question text, so `prepare_listen.py` splits the two tasks by matching
`question` against the fixed prompt lists in
`configs/data/listen_label_map.yaml`. It also drops samples whose `id` names
CREMA-D or MELD (both are in our fine-tuning data) and normalises each task's
labels.

The two tasks use different label spaces:

- **Acoustic** keeps the eight categories the LALM prompt offers — happy, sad,
  angry, fear, disgust, surprise, neutral, calm — so the model is scored on the
  label space it was asked to choose from, not on CREMA-ASIS's five. Corpus
  spelling differences (`happiness`, `anger`, `fearful`, …) are normalised by
  `acoustic_synonyms`, which is applied to the model's output as well.
- **Semantic** collapses LISTEN's emotion words onto positive / negative /
  neutral via `semantic_map`, since the model predicts a polarity.

Everything is in the config so both can be inspected and edited.

```bash
# inspect the schema and label distribution first
python scripts/prepare_listen.py \
    --parquet data/LISTEN/test-00000-of-00001.parquet --inspect

python scripts/prepare_listen.py \
    --parquet data/LISTEN/test-00000-of-00001.parquet \
    --output-dir data/LISTEN
```

This prints how many samples survive each stage and lists every label it could
not place, so out-of-vocabulary categories are visible rather than silently
dropped.

Then evaluate each subset:

```bash
python scripts/evaluate.py \
    --model qwen2-audio \
    --model-config configs/models/qwen2_audio.yaml \
    --lora-path finetuned_models/qwen2-audio-lora/checkpoint \
    --data data/LISTEN/listen_acoustic.parquet \
    --data-format parquet \
    --temp-audio-dir LISTEN_audios \
    --output results/qwen2_lora_listen_acoustic.csv

python scripts/compute_metrics.py \
    --predictions results/qwen2_lora_listen_acoustic.csv \
    --single-task --true-column acoustic --pred-column acoustic_emotion \
    --label-map configs/data/listen_label_map.yaml \
    --label-map-key acoustic_synonyms
```

Samples the model failed to process keep an empty prediction and are counted as
incorrect, so the denominator is the full filtered subset.

---

### 5. Embedding extraction

Extracts and caches layer-wise embeddings for the full dataset. Embeddings are saved under `embedding_cache/` by default, organized by model and component.

**Qwen2-Audio — LLM layers**
```bash
python scripts/extract_embeddings.py \
    --model qwen2-audio --component llm \
    --model-config configs/models/qwen2_audio.yaml \
    --csv data/CREMA-ASIS_meta.csv \
    --data-dir data/crema-asis/cremad-sync-wsad
```

**Qwen2-Audio — Whisper encoder + multi-modal projector**
```bash
python scripts/extract_embeddings.py \
    --model qwen2-audio --component whisper \
    --model-config configs/models/qwen2_audio.yaml \
    --csv data/CREMA-ASIS_meta.csv \
    --data-dir data/crema-asis/cremad-sync-wsad
```

**Qwen2-Audio — fine-tuned LLM**
```bash
python scripts/extract_embeddings.py \
    --model qwen2-audio --component llm \
    --model-config configs/models/qwen2_audio.yaml \
    --lora-path finetuned_models/qwen2-audio-lora/checkpoint \
    --csv data/CREMA-ASIS_meta.csv \
    --data-dir data/crema-asis/cremad-sync-wsad
```

**Audio-Flamingo3 — base**
```bash
python scripts/extract_embeddings.py \
    --model audio-flamingo3 --component llm \
    --model-config configs/models/audio_flamingo3.yaml \
    --csv data/CREMA-ASIS_meta.csv \
    --data-dir data/crema-asis/cremad-sync-wsad
```

**Audio-Flamingo3 — fine-tuned**
```bash
python scripts/extract_embeddings.py \
    --model audio-flamingo3 --component llm \
    --model-config configs/models/audio_flamingo3.yaml \
    --lora-path finetuned_models/audio-flamingo3-lora/checkpoint \
    --csv data/CREMA-ASIS_meta.csv \
    --data-dir data/crema-asis/cremad-sync-wsad
```

**Kimi-Audio — base**
```bash
python scripts/extract_embeddings.py \
    --model kimi-audio --component llm \
    --model-config configs/models/kimi_audio.yaml \
    --csv data/CREMA-ASIS_meta.csv \
    --data-dir data/crema-asis/cremad-sync-wsad
```

**Kimi-Audio — fine-tuned**
```bash
python scripts/extract_embeddings.py \
    --model kimi-audio --component llm \
    --model-config configs/models/kimi_audio.yaml \
    --lora-path finetuned_models/kimi-audio-lora/checkpoint \
    --csv data/CREMA-ASIS_meta.csv \
    --data-dir data/crema-asis/cremad-sync-wsad
```

---

### 6. Linear probing

Trains linear probes on the cached embeddings. The `--cache-dir` should point to wherever `extract_embeddings.py` saved its output (defaults to `embedding_cache/`).

**Qwen2-Audio — LLM layers**
```bash
python scripts/run_probing.py \
    --model qwen2-audio --component llm \
    --model-config configs/models/qwen2_audio.yaml \
    --probe-config configs/probing/default.yaml \
    --csv data/CREMA-ASIS_meta.csv \
    --data-dir data/crema-asis/cremad-sync-wsad \
    --cache-dir embedding_cache \
    --results-dir probe_results
```

**Qwen2-Audio — Whisper encoder + multi-modal projector**
```bash
python scripts/run_probing.py \
    --model qwen2-audio --component whisper \
    --model-config configs/models/qwen2_audio.yaml \
    --probe-config configs/probing/default.yaml \
    --csv data/CREMA-ASIS_meta.csv \
    --data-dir data/crema-asis/cremad-sync-wsad \
    --cache-dir embedding_cache \
    --results-dir probe_results
```

**Qwen2-Audio — fine-tuned**
```bash
python scripts/run_probing.py \
    --model qwen2-audio --component llm \
    --model-config configs/models/qwen2_audio.yaml \
    --probe-config configs/probing/default.yaml \
    --lora-path finetuned_models/qwen2-audio-lora/checkpoint \
    --csv data/CREMA-ASIS_meta.csv \
    --data-dir data/crema-asis/cremad-sync-wsad \
    --cache-dir embedding_cache \
    --results-dir probe_results
```

**Audio-Flamingo3 — base**
```bash
python scripts/run_probing.py \
    --model audio-flamingo3 --component llm \
    --model-config configs/models/audio_flamingo3.yaml \
    --probe-config configs/probing/default.yaml \
    --csv data/CREMA-ASIS_meta.csv \
    --data-dir data/crema-asis/cremad-sync-wsad \
    --cache-dir embedding_cache \
    --results-dir probe_results
```

**Audio-Flamingo3 — fine-tuned**
```bash
python scripts/run_probing.py \
    --model audio-flamingo3 --component llm \
    --model-config configs/models/audio_flamingo3.yaml \
    --probe-config configs/probing/default.yaml \
    --lora-path finetuned_models/audio-flamingo3-lora/checkpoint \
    --csv data/CREMA-ASIS_meta.csv \
    --data-dir data/crema-asis/cremad-sync-wsad \
    --cache-dir embedding_cache \
    --results-dir probe_results
```

**Kimi-Audio — base**
```bash
python scripts/run_probing.py \
    --model kimi-audio --component llm \
    --model-config configs/models/kimi_audio.yaml \
    --probe-config configs/probing/default.yaml \
    --csv data/CREMA-ASIS_meta.csv \
    --data-dir data/crema-asis/cremad-sync-wsad \
    --cache-dir embedding_cache \
    --results-dir probe_results
```

**Kimi-Audio — fine-tuned**
```bash
python scripts/run_probing.py \
    --model kimi-audio --component llm \
    --model-config configs/models/kimi_audio.yaml \
    --probe-config configs/probing/default.yaml \
    --lora-path finetuned_models/kimi-audio-lora/checkpoint \
    --csv data/CREMA-ASIS_meta.csv \
    --data-dir data/crema-asis/cremad-sync-wsad \
    --cache-dir embedding_cache \
    --results-dir probe_results
```

## Model Weights For Reproducability
To facilitate reproducibility, we provide our fine-tuned LoRA weights under `weights_used_for_study/` 




## TODOs
* **TODO**: Unify qwen2-audio LLM-layer extraction with Audio-Flamingo3 and Kimi-Audio by using forward hooks instead of relying on output embeddings from transformers. Reduces the forward pass of the audio encoder by half. (The Whisper/projector path already runs the audio tower once and reuses its output.)


## Citation
