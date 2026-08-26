"""GPT-4o filtering of candidate GoEmotions sentences.

Implements the sentence-selection stage. A sentence is rejected when the model answers
``True``, i.e. when at least one of these holds:

1. it does not express the target sentiment,
2. it is strongly racist, very offensive, or inappropriate for public use,
3. it only exists in written form and would not be said aloud.

The response format is ``"False"`` or ``"True. <comma-separated reasons>"``.
"""

from __future__ import annotations

import os
import re
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm


DEFAULT_MODEL = "gpt-4o"


def build_prompt(template: str, sentence: str, sentiment: str) -> str:
    """Fill the sentence-selection template for one sentence."""
    return template.format(sentiment=sentiment, sentence=sentence)


def parse_decision(response: str) -> Tuple[bool, List[int]]:
    """Parse a raw model reply into ``(rejected, reasons)``.

    Args:
        response: Raw text, e.g. ``"True. 1,3"`` or ``"False"``.

    Returns:
        ``rejected`` is ``True`` when the sentence should be dropped.
        ``reasons`` holds the criterion numbers the model cited.
    """
    text = str(response).strip()
    rejected = text.lower().lstrip("*").startswith("true")
    reasons = [int(n) for n in re.findall(r"\b([123])\b", text)] if rejected else []
    return rejected, sorted(set(reasons))


def _default_client():
    """Create an OpenAI client from ``OPENAI_API_KEY``."""
    from openai import OpenAI

    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def filter_sentences(
    df: pd.DataFrame,
    template: str,
    sentence_column: str = "sentence",
    sentiment_column: Optional[str] = "sentiment",
    sentiment: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    client=None,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Run the GPT filter over every row of *df*.

    Args:
        df: Candidate sentences.
        template: Prompt template with ``{sentiment}`` / ``{sentence}`` slots.
        sentence_column: Column holding the sentence text.
        sentiment_column: Column holding the target sentiment per row.
        sentiment: Fixed target sentiment, used when *sentiment_column* is None.
        model: OpenAI model name.
        client: Pre-built OpenAI client; created from the environment if None.
        max_retries: Retries per sentence on API errors.

    Returns:
        A copy of *df* with ``gpt_raw``, ``gpt_rejected``, ``gpt_reasons``, and
        ``keep`` columns.
    """
    if client is None:
        client = _default_client()

    raws: List[str] = []
    rejects: List[Optional[bool]] = []
    reason_strs: List[str] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="GPT sentence filter"):
        target = sentiment if sentiment_column is None else row[sentiment_column]
        prompt = build_prompt(template, row[sentence_column], target)

        reply = None
        for attempt in range(max_retries):
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                reply = completion.choices[0].message.content
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Giving up on {row[sentence_column]!r}: {e}")
                else:
                    time.sleep(2 ** attempt)

        if reply is None:
            raws.append("")
            rejects.append(None)
            reason_strs.append("")
            continue

        rejected, reasons = parse_decision(reply)
        raws.append(reply)
        rejects.append(rejected)
        reason_strs.append(",".join(str(r) for r in reasons))

    out = df.copy()
    out["gpt_raw"] = raws
    out["gpt_rejected"] = rejects
    out["gpt_reasons"] = reason_strs
    # Rows the API never answered for are kept, so nothing is silently dropped.
    out["keep"] = out["gpt_rejected"] != True  # noqa: E712
    return out
