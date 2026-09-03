"""Scout (localizer) body-habitus measurement.

Deterministic body width/depth per vertebral level from the CT scout films --
the only series that contains the whole body outline on a spine protocol.
"""
from .loader import ScoutImage, find_scouts, load_scout
from .thickness import (body_profile, measure_habitus, level_z_ranges,
                        habitus_for_case, scout_json)

__all__ = ["ScoutImage", "find_scouts", "load_scout", "body_profile",
           "measure_habitus", "level_z_ranges", "habitus_for_case",
           "scout_json"]
