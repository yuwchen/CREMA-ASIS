"""Text parsing utilities for model outputs and filenames."""

import ast
import json
import os
import re


def auto_parse(s: str):
    """Attempt to parse a string as JSON or Python literal.

    Tries multiple strategies in order:
    1. Standard ``json.loads``
    2. Single-quote fix then ``json.loads``
    3. ``ast.literal_eval``

    Args:
        s: Raw string from model output.

    Returns:
        Parsed dict/list or ``None`` on failure.
    """
    if isinstance(s, dict):
        return s

    try:
        return json.loads(s)
    except Exception:
        pass

    try:
        fixed = s.replace("'", '"')
        fixed = re.sub(r'(?<=\w)"(?=\w)', "'", fixed)
        return json.loads(fixed)
    except Exception:
        pass

    try:
        return ast.literal_eval(s)
    except Exception as e:
        print(f"Failed to parse: {e}")
        return None


def clean_response(response: str) -> str:
    """Clean a raw model response string before JSON parsing.

    Strips escaped quotes, angle-bracket artifacts, and partial braces
    that commonly appear in LLM outputs.

    Args:
        response: Raw text from model generation.

    Returns:
        Cleaned string ready for ``auto_parse`` or ``json.loads``.
    """
    response = response.replace('\\"', '"')
    if "<" in response or ">" in response:
        response = response.split("<")[0]
    if "{" not in response and "}" in response:
        response = response.split("}")[0].strip()
    return response


def parse_emotion_response(response: str) -> dict:
    """Parse model response into acoustic_emotion / semantic_sentiment dict.

    Applies ``clean_response`` then ``auto_parse``.  Falls back to
    comma-splitting when JSON parsing fails entirely.

    Args:
        response: Raw model output string.

    Returns:
        Dict with ``acoustic_emotion`` and ``semantic_sentiment`` keys.
    """
    cleaned = clean_response(response)
    result = auto_parse(cleaned)
    if result is not None:
        return result

    # Fallback: comma-split heuristic
    chunks = cleaned.split(",")
    if len(chunks) > 1:
        return {
            "acoustic_emotion": chunks[0].strip(),
            "semantic_sentiment": chunks[1].strip(),
        }
    return {
        "acoustic_emotion": chunks[0].strip() if chunks else "",
        "semantic_sentiment": "",
    }


def clean_sentence(sentence: str) -> str:
    """Remove bracketed and angle-bracketed content from a sentence."""
    cleaned = re.sub(r"\[.*?\]", "", sentence)
    cleaned = re.sub(r"\<.*?\>", "", cleaned)
    return cleaned


def extract_parentheses(text: str) -> str:
    """Extract content inside first pair of parentheses, or return original."""
    match = re.search(r"\((.*?)\)", text)
    return match.group(1) if match else text


def parse_filename(filename: str):
    """Parse filename to extract acoustic emotion and semantic label.

    Expected format: ``ORIG_NAME-ACOUSTIC_EMOTION-SEMANTIC_LABEL-IDX.wav``
    Example: ``1008_IWL_NEU_XX-neutral-positive-598.wav``

    Returns:
        Tuple of ``(acoustic_emotion, semantic_label)``.

    Raises:
        ValueError: If filename does not match expected format.
    """
    basename = os.path.basename(filename).replace(".wav", "")
    parts = basename.split("-")
    if len(parts) >= 4:
        return parts[-3], parts[-2]
    raise ValueError(f"Filename {filename} doesn't match expected format")
