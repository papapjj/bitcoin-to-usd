"""Compatibility name for artifacts written before TimingNet became canonical.

New code must not import or instantiate this class.  It is intentionally empty
and exists only because Python pickle stores the original module/class path in
older v13 artifacts; :class:`BoNhanDienNguCanh` removes the deserialized field
during ``__setstate__``.  All live timing signals now come from
``self_growing_trader.timing_network.TimingNetwork``.
"""

from __future__ import annotations


class TimingDetector:  # pragma: no cover - pickle migration symbol only
    """Dead migration symbol; it contains no timing logic."""

    pass


__all__ = ["TimingDetector"]
