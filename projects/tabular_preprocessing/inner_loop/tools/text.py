"""Shared formatting for tool output: short, plain text an agent can read."""

from __future__ import annotations

from typing import Any

import pandas as pd

MAX_LIST = 40  # longest list a tool prints before it says how many it left out


def fmt(value: Any) -> str:
    """A value as short text: numbers with 4 significant digits, long text cut."""
    if isinstance(value, float):
        return "nan" if pd.isna(value) else f"{value:.4g}"
    text = str(value)
    return text if len(text) <= 40 else text[:37] + "..."


def pct(share: float) -> str:
    """A share as a percentage."""
    return f"{100 * share:.1f}%"


def bullets(lines: list[str], limit: int = MAX_LIST) -> str:
    """Lines as a bullet list, cut to ``limit`` with a note on how many were left out."""
    shown = [f"- {line}" for line in lines[:limit]]
    if len(lines) > limit:
        shown.append(f"- ... and {len(lines) - limit} more")
    return "\n".join(shown) if shown else "- (none)"


def table(df: pd.DataFrame, rows: int = 10) -> str:
    """A frame as aligned text, at most ``rows`` rows, long cells cut."""
    return df.head(rows).map(fmt).to_string(index=False)
