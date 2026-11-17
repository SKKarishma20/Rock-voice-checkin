"""Checklist schema — the single source of truth for what we must collect.

Start small: 3 fields. The Pydantic model owns validation + coercion, so the
LLM can pass loose spoken values ("nope", "a seven") and we either store a
clean value or hand back a short error the agent can re-ask. Add fields here
later — the agent code reads FIELDS/ORDER, so nothing else changes.

`validate_assignment=True` means `setattr(case, field, value)` validates that
one field and raises ValidationError on bad input — perfect for one-at-a-time
voice collection.
"""

from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel, ConfigDict, field_validator

# Ordered: (field_name, spoken question). ORDER drives "what's next".
FIELDS: list[tuple[str, str]] = [
    ("took_medication", "Did you take your medication today?"),
    ("pain_level", "On a scale of one to ten, how would you rate your pain right now?"),
    ("needs_help", "Is there anything you need help with before our next check-in?"),
]
ORDER: list[str] = [name for name, _ in FIELDS]
QUESTION: dict[str, str] = dict(FIELDS)

_YES = {"yes", "y", "yeah", "yep", "yup", "true", "did", "i did", "took", "correct", "sure", "affirmative"}
_NO = {"no", "n", "nope", "nah", "false", "didn't", "did not", "haven't", "have not", "skip", "skipped", "forgot"}
_NUM_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


class CheckinCase(BaseModel):
    """Per-call state. All fields start None (= still missing)."""

    model_config = ConfigDict(validate_assignment=True)

    took_medication: Optional[bool] = None
    pain_level: Optional[int] = None
    needs_help: Optional[str] = None

    @field_validator("took_medication", mode="before")
    @classmethod
    def _coerce_bool(cls, v):
        if v is None or isinstance(v, bool):
            return v
        s = str(v).strip().lower().rstrip(".!")
        if s in _YES:
            return True
        if s in _NO:
            return False
        raise ValueError("please answer yes or no")

    @field_validator("pain_level", mode="before")
    @classmethod
    def _coerce_pain(cls, v):
        if v is None or isinstance(v, int):
            return v
        s = str(v).strip().lower()
        m = re.search(r"\d+", s)
        if m:
            return int(m.group())
        # a number word anywhere in the phrase ("a seven", "about three")
        for token in re.findall(r"[a-z]+", s):
            if token in _NUM_WORDS:
                return _NUM_WORDS[token]
        raise ValueError("give a number from 0 to 10")

    @field_validator("pain_level")
    @classmethod
    def _range_pain(cls, v):
        if v is not None and not (0 <= v <= 10):
            raise ValueError("pain must be between 0 and 10")
        return v

    @field_validator("needs_help", mode="before")
    @classmethod
    def _clean_help(cls, v):
        if v is None:
            return v
        s = str(v).strip()
        if not s:
            raise ValueError("tell me what you need help with, or say nothing")
        return s[:300]
