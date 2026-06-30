"""Validation without external ground truth: internal consistency + literature."""
from .consistency import (scan_rescan, mode_agreement, literature_band,
                          generate_report)

__all__ = ["scan_rescan", "mode_agreement", "literature_band", "generate_report"]
