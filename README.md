# CREMA-ASIS

This repo includes the code related to the study
When Vocal Tone and Literal Meaning Diverge: An Acoustic–Semantic Incongruity Study for Large Audio–Language Models



# Notes
## TODOs
* **TODO**: Unify qwen2-audio feature extraction to be similar to Audio-Flamingo3 and Kimi-Audio by using forward hooks instead of relying on output embeddings from transformers. Reduces the forward pass of the audio encoder by half.
* **TODO**

# Setup
Different models require different transformer versions.
Qwen2-audio:
```
python -m pip install transformers==4.46.1
```

Audio Flamingo
```
python -m pip install --upgrade git+https://github.com/huggingface/transformers accelerate
```

Kimi-Audio

Use the kimi-audio setup.
See `https://github.com/MoonshotAI/Kimi-Audio/tree/master?tab=readme-ov-file#getting-started`

Run with deepspeed to enable multi-GPU training.