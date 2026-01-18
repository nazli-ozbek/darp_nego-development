"""I/O helpers for scenario metrics."""

from __future__ import annotations

import json
from typing import Dict, Optional


def load_company_cases(file_path: str, case_id: Optional[str] = None) -> Dict:
    """Load company cases from a JSON file with optional case filtering."""
    with open(file_path, "r") as handle:
        data = json.load(handle)

    if case_id is None:
        return data

    if case_id not in data:
        raise KeyError(f"Case '{case_id}' not found in {file_path}")
    return {case_id: data[case_id]}
