#!/usr/bin/env python3
"""Definitive stage unavailable: fail explicitly instead of succeeding empty."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.atlas_contract import reject_legacy

if __name__ == '__main__':
    reject_legacy(Path(__file__).name)
