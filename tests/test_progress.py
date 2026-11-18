"""The status line long commands print (`progress.py`), written to a pipe as a log or CI sees it."""

from __future__ import annotations

import io

from surrogate.progress import Progress, Step


def test_piped_status_is_plain_and_estimates_only_from_one_percent() -> None:
    """Catches carriage returns or escape codes in a log, a time left extrapolated from the first moments of
    a run (a fraction of a second of work putting an hour-long run at days), and a network shown with the
    held-out runs of the one before it once thirty epochs of 1/30 have finished it."""
    out = io.StringIO()
    with Progress([Step("Shipped networks", 5, "network", 50), Step("Learning curve", 55, "network", 550)], out) as p:
        p.step("Shipped networks")
        p.describe("runs 80 and 98 held out")
        p.advance(1 / 30)
        early = p.status()
        for _ in range(29):
            p.advance(1 / 30)
        between = p.status()
        p.note("Shipped: CV R² 0.9458")
        p.step("Learning curve")
    lines = out.getvalue().splitlines()
    assert not any("\r" in line or "\x1b" in line for line in lines)
    assert len(lines) == 3 and lines[1] == "Shipped: CV R² 0.9458"
    assert lines[0].endswith("Shipped networks: network 1 of 5")
    assert lines[2].endswith("Learning curve: network 1 of 55")
    assert "left" not in early and "runs 80 and 98" in early
    assert "left" in between and between.endswith("Shipped networks: network 2 of 5")
