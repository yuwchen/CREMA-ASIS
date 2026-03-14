"""Unified LoRA fine-tuning entry point.

Consolidates ``qwen_finetune.py``, ``flamingo_finetune.py``, and
``kimi_training.py`` into a single configurable pipeline.

The model-specific differences (collator, trainable components, PEFT
preparation) are handled by the :class:`~src.models.base.AudioModel`
interface.  For Kimi-Audio, a custom :class:`KimiAudioTrainer` sub-class
overrides the loss computation.
"""

from __future__ import annotations

from typing import List, Optional

import torch
from peft import get_peft_model
from transformers import Trainer, TrainingArguments

from src.utils.io import load_yaml, load_prompt, ensure_dir
from src.models import get_model

# ---------------------------------------------------------------------------
# Kimi-specific trainer (custom forward / loss)
# ---------------------------------------------------------------------------

class KimiAudioTrainer(Trainer):
    """Custom trainer for Kimi-Audio with proper forward pass."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids = inputs.pop("input_ids")
        text_input_ids = inputs.pop("text_input_ids")
        is_continuous_mask = inputs.pop("is_continuous_mask")
        whisper_input_feature = inputs.pop("whisper_input_feature")
        labels = inputs.pop("labels")

        outputs = model(
            input_ids=input_ids,
            text_input_ids=text_input_ids,
            whisper_input_feature=whisper_input_feature,
            is_continuous_mask=is_continuous_mask,
            return_dict=False,
        )
        _audio_logits, text_logits, _past_kv = outputs

        loss_fct = torch.nn.CrossEntropyLoss()
        shift_logits = text_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Main fine-tuning function
# ---------------------------------------------------------------------------

def run_finetuning(
    model_type: str,
    model_config_path: str,
    train_config_path: str,
    train_dataset: list,
    val_dataset: list,
    output_dir: str,
    prompt_path: str = "configs/prompts/emotion_sentiment.txt",
    run_name: Optional[str] = None,
) -> None:
    """Run LoRA fine-tuning for any supported model.

    Args:
        model_type: One of ``"qwen2-audio"``, ``"kimi-audio"``, ``"audio-flamingo3"``.
        model_config_path: Path to model YAML config.
        train_config_path: Path to training YAML config.
        train_dataset: List of ``{"audio_path": ..., "target": ...}`` dicts,
            or for Kimi a list of dicts with ``chat`` / ``filepath`` / etc.
        val_dataset: Same format as *train_dataset*.
        output_dir: Directory for checkpoints and logs.
        prompt_path: Path to the task prompt text file.
        run_name: Optional W&B / TensorBoard run name.
    """

    model_cfg = load_yaml(model_config_path)
    train_cfg = load_yaml(train_config_path)
    task_prompt = load_prompt(prompt_path)

    # -- Load model --
    audio_model = get_model(model_type, model_name=model_cfg["model_name"])
    audio_model.load(device="auto")

    # -- Kimi-specific preparation --
    if model_type == "kimi-audio":
        audio_model.prepare_for_training()

    # -- Apply LoRA --
    lora_config = audio_model.get_lora_config(model_cfg)
    trainable = audio_model.get_trainable_model()
    trainable = get_peft_model(trainable, lora_config)
    trainable.print_trainable_parameters()

    # Store back (important for Kimi where ALM is the trainable part)
    if model_type == "kimi-audio":
        audio_model.model.alm = trainable
    elif model_type in ("qwen2-audio", "audio-flamingo3"):
        audio_model.model = trainable

    # -- Collator --
    collator = audio_model.get_collator(task_prompt)

    # -- Training arguments --
    ensure_dir(output_dir)
    if run_name is None:
        run_name = f"{model_type}-lr_{train_cfg['learning_rate']}-ep_{train_cfg['num_epochs']}"

    training_args = TrainingArguments(
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["batch_size"],
        per_device_eval_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        logging_steps=train_cfg["logging_steps"],
        output_dir=output_dir,
        eval_strategy=train_cfg["eval_strategy"],
        save_steps=train_cfg["save_steps"],
        eval_steps=train_cfg["eval_steps"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        bf16=train_cfg.get("bf16", False),
        remove_unused_columns=False,
        report_to=train_cfg["report_to"],
        run_name=run_name,
        logging_dir=f"./logs/{output_dir}",
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs=train_cfg.get(
            "gradient_checkpointing_kwargs", {"use_reentrant": False}
        ),
    )

    # -- Trainer --
    TrainerClass = KimiAudioTrainer if model_type == "kimi-audio" else Trainer
    trainer = TrainerClass(
        model=trainable,
        args=training_args,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # -- Train --
    print(f"\nStarting {model_type} fine-tuning → {output_dir}")
    trainer.train()

    # -- Save --
    trainable.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")
