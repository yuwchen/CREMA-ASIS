# CREMA-ASIS

Code for the study *When Vocal Tone and Literal Meaning Diverge: An Acoustic–Semantic Incongruity Study for Large Audio–Language Models*.

The project investigates whether large audio-language models (LALMs) can detect incongruity between a speaker's emotional tone and the literal meaning of what they say.

We introduce CREMA-ASIS, a dataset of ~77k synthetic utterances where acoustic emotion and semantic sentiment are independently controlled, covering both incongruous (e.g., angry voice saying something positive) and congruous conditions. 

Using this dataset, we evaluate three models: (1) Qwen2-Audio, (2) Audio-Flamingo3, and (3) Kimi-Audio, both off-the-shelf and after LoRA fine-tuning. We also perform linear probing of internal representations to understand how these models process acoustic vs. semantic information.

---

## Project structure

```
CREMA-ASIS/
├── configs/
│   ├── models/          # per-model YAML configs
│   ├── probing/         # probing hyperparameters
│   ├── prompts/         # prompt templates
│   └── training/        # fine-tuning configs
├── data/                # put your data here (see below)
├── scripts/             # entry points for each pipeline stage
├── src/                 # library code
├── finetuned_models/    # LoRA checkpoints (written by finetune.py)
├── embedding_cache/     # layer embeddings (written by extract_embeddings.py)
├── probe_results/       # probing outputs (written by run_probing.py)
└── results/             # evaluation outputs (written by evaluate.py)
```

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
├── cremad-sync/
│   └── cremad-sync-wsad/         # CREMA-ASIS generated audio (WAV)
├── CREMA-D/
│   └── AudioWAV_en/              # original CREMA-D audio
├── cremad_all_clean_w_sad_filtered.csv   # CREMA-ASIS full split (embeddings/probing/fine-tuning)
├── crema_d_test.csv                      # CREMA-ASIS test split (evaluation)
└── crema-d_en_split.csv                  # original CREMA-D split (fine-tuning)
```

If you also want to fine-tune on MELD, place it at `data/MELD.Raw/` following the expected directory layout (`meld_train.csv`, `meld_val.csv`, `train_wav/`, `dev_wav/`).

---

## Pipeline

The pipeline runs in roughly this order:

1. Generate synthetic audio
2. Filter by acoustic quality
3. Fine-tune
4. Evaluate
5. Extract embeddings
6. Run probing

---

### 1. Data generation

The CREMA-ASIS dataset is built using IndexTTS2 TTS with voice cloning. The idea is to generate speech where we independently control the acoustic emotion (via a CREMA-D reference clip that carries the target emotion) and the semantic content (via a sentence from GoEmotions). This lets us create utterances where the two are incongruous — e.g., an angry-sounding voice saying something positive — at a scale that can't be obtained from natural data. After generation, an acoustic emotion recognition (AER) model is used to filter out samples where the intended emotion clearly wasn't conveyed (see step 2).

For a single sample:
```bash
python scripts/generate_data.py \
    --cfg checkpoints/config.yaml \
    --model-dir checkpoints \
    --speaker data/cremad-sync/cremad-sync-wsad/1001_DFA_ANG_XX.wav \
    --text "I appreciate it, that's good to know." \
    --output data/samples/output.wav \
    --emotion 0 0 0 0 0 0 0 0
```

For batch generation from a CSV manifest (recommended):

The CSV should have columns: `audio_path`, `output_text`, `emo_vector` (8-dim list), `output_name`.

```bash
python scripts/generate_data_from_csv.py \
    --cfg checkpoints/config.yaml \
    --model-dir checkpoints \
    --speaker-dir data/cremad-sync/cremad-sync-wsad \
    --csv CREMA-ASIS_meta.csv \
    --output-dir data/samples \
    --skip-existing
```

---

### 2. Data filtering

After generation, run an acoustic emotion recognition (AER) model over the output to identify samples where the intended emotion clearly wasn't reproduced. The AER uses a Whisper-large-v3-based classifier by default.

```bash
python scripts/filter_data.py \
    --wav-dir data/samples \
    --output-csv data/acoustic_detection.csv
```

This writes a CSV with per-file predicted emotion labels.
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
deepspeed --num_gpus=2 scripts/finetune.py \
    --model kimi-audio \
    --model-config configs/models/kimi_audio.yaml \
    --train-config configs/training/training_kimi_audio.yaml \
    --output-dir finetuned_models/kimi-audio-lora \
    --datasets cremad_annotated cremad_base meld
```

---

### 4. Evaluation

Evaluate a model on the test set. Run without `--lora-path` for the base model, or with it for a fine-tuned checkpoint.

**Qwen2-Audio (base)**
```bash
python scripts/evaluate.py \
    --model qwen2-audio \
    --model-config configs/models/qwen2_audio.yaml \
    --data data/crema_d_test.csv \
    --output results/qwen2_base.csv
```

**Qwen2-Audio (fine-tuned)**
```bash
python scripts/evaluate.py \
    --model qwen2-audio \
    --model-config configs/models/qwen2_audio.yaml \
    --lora-path finetuned_models/qwen2-audio-lora/checkpoint \
    --data data/crema_d_test.csv \
    --output results/qwen2_lora.csv
```

**Audio-Flamingo3 (base)**
```bash
python scripts/evaluate.py \
    --model audio-flamingo3 \
    --model-config configs/models/audio_flamingo3.yaml \
    --data data/crema_d_test.csv \
    --output results/audio_flamingo3_base.csv
```

**Kimi-Audio (base)**
```bash
python scripts/evaluate.py \
    --model kimi-audio \
    --model-config configs/models/kimi_audio.yaml \
    --data data/crema_d_test.csv \
    --output results/kimi_audio_base.csv
```

---

### 5. Embedding extraction

Extracts and caches layer-wise embeddings for the full dataset. Embeddings are saved under `embedding_cache/` by default, organized by model and component.

**Qwen2-Audio — LLM layers**
```bash
python scripts/extract_embeddings.py \
    --model qwen2-audio --component llm \
    --model-config configs/models/qwen2_audio.yaml \
    --csv data/cremad_all_clean_w_sad_filtered.csv \
    --data-dir data/cremad-sync/cremad-sync-wsad
```

**Qwen2-Audio — Whisper encoder**
```bash
python scripts/extract_embeddings.py \
    --model qwen2-audio --component whisper \
    --model-config configs/models/qwen2_audio.yaml \
    --csv data/cremad_all_clean_w_sad_filtered.csv \
    --data-dir data/cremad-sync/cremad-sync-wsad
```

**Qwen2-Audio — fine-tuned LLM**
```bash
python scripts/extract_embeddings.py \
    --model qwen2-audio --component llm \
    --model-config configs/models/qwen2_audio.yaml \
    --lora-path finetuned_models/qwen2-audio-lora/checkpoint \
    --csv data/cremad_all_clean_w_sad_filtered.csv \
    --data-dir data/cremad-sync/cremad-sync-wsad
```

**Audio-Flamingo3 — base**
```bash
python scripts/extract_embeddings.py \
    --model audio-flamingo3 --component llm \
    --model-config configs/models/audio_flamingo3.yaml \
    --csv data/cremad_all_clean_w_sad_filtered.csv \
    --data-dir data/cremad-sync/cremad-sync-wsad
```

**Audio-Flamingo3 — fine-tuned**
```bash
python scripts/extract_embeddings.py \
    --model audio-flamingo3 --component llm \
    --model-config configs/models/audio_flamingo3.yaml \
    --lora-path finetuned_models/audio-flamingo3-lora/checkpoint \
    --csv data/cremad_all_clean_w_sad_filtered.csv \
    --data-dir data/cremad-sync/cremad-sync-wsad
```

**Kimi-Audio — base**
```bash
python scripts/extract_embeddings.py \
    --model kimi-audio --component llm \
    --model-config configs/models/kimi_audio.yaml \
    --csv data/cremad_all_clean_w_sad_filtered.csv \
    --data-dir data/cremad-sync/cremad-sync-wsad
```

**Kimi-Audio — fine-tuned**
```bash
python scripts/extract_embeddings.py \
    --model kimi-audio --component llm \
    --model-config configs/models/kimi_audio.yaml \
    --lora-path finetuned_models/kimi-audio-lora/checkpoint \
    --csv data/cremad_all_clean_w_sad_filtered.csv \
    --data-dir data/cremad-sync/cremad-sync-wsad
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
    --csv data/cremad_all_clean_w_sad_filtered.csv \
    --data-dir data/cremad-sync/cremad-sync-wsad \
    --cache-dir embedding_cache \
    --results-dir probe_results
```

**Qwen2-Audio — Whisper encoder**
```bash
python scripts/run_probing.py \
    --model qwen2-audio --component whisper \
    --model-config configs/models/qwen2_audio.yaml \
    --probe-config configs/probing/default.yaml \
    --csv data/cremad_all_clean_w_sad_filtered.csv \
    --data-dir data/cremad-sync/cremad-sync-wsad \
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
    --csv data/cremad_all_clean_w_sad_filtered.csv \
    --data-dir data/cremad-sync/cremad-sync-wsad \
    --cache-dir embedding_cache \
    --results-dir probe_results
```

**Audio-Flamingo3 — base**
```bash
python scripts/run_probing.py \
    --model audio-flamingo3 --component llm \
    --model-config configs/models/audio_flamingo3.yaml \
    --probe-config configs/probing/default.yaml \
    --csv data/cremad_all_clean_w_sad_filtered.csv \
    --data-dir data/cremad-sync/cremad-sync-wsad \
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
    --csv data/cremad_all_clean_w_sad_filtered.csv \
    --data-dir data/cremad-sync/cremad-sync-wsad \
    --cache-dir embedding_cache \
    --results-dir probe_results
```

**Kimi-Audio — base**
```bash
python scripts/run_probing.py \
    --model kimi-audio --component llm \
    --model-config configs/models/kimi_audio.yaml \
    --probe-config configs/probing/default.yaml \
    --csv data/cremad_all_clean_w_sad_filtered.csv \
    --data-dir data/cremad-sync/cremad-sync-wsad \
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
    --csv data/cremad_all_clean_w_sad_filtered.csv \
    --data-dir data/cremad-sync/cremad-sync-wsad \
    --cache-dir embedding_cache \
    --results-dir probe_results
```

## Model Weights For Reproducability
To facilitate reproducibility, we provide our fine-tuned LoRA weights under `weights_used_for_study/` 



## TODOs
* **TODO**: Unify qwen2-audio feature extraction to be similar to Audio-Flamingo3 and Kimi-Audio by using forward hooks instead of relying on output embeddings from transformers. Reduces the forward pass of the audio encoder by half.