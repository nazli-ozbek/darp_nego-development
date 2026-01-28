"""I/O helpers for scenario metrics."""

from __future__ import annotations

import json
from typing import Dict, Iterable, Optional, List


def load_company_cases(
    file_path: str,
    case_id: Optional[str] = None,
    case_ids: Optional[Iterable[str]] = None,
) -> Dict:
    """Load company cases from a JSON file with optional case filtering."""
    with open(file_path, "r") as handle:
        data = json.load(handle)

    selected: Optional[List[str]] = None
    if case_ids:
        selected = list(case_ids)
    elif case_id:
        selected = [case_id]

    if not selected:
        return data

    missing = [cid for cid in selected if cid not in data]
    if missing:
        raise KeyError(f"Case(s) {missing} not found in {file_path}")
    return {cid: data[cid] for cid in selected}
