"""Utility modules shared across data-prep and HAS scripts."""

from __future__ import annotations

import sys

# Some Toucan dumps contain extremely large numeric literals (e.g. long IDs).
# Python 3.11+ protects against pathological ints via sys.set_int_max_str_digits.
# Relax the guard globally so json.loads can parse those payloads.
if hasattr(sys, "set_int_max_str_digits"):
    try:
        sys.set_int_max_str_digits(0)
    except (ValueError, TypeError):
        # Either unsupported interpreter or already configured elsewhere.
        pass


