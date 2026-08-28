"""Generate a deterministic, clearly synthetic dataset for the CourtVision demo."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def main(rows: int = 160, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    idx = np.arange(rows)
    age = rng.uniform(18.0, 23.5, rows).round(1)
    height = rng.normal(78, 3.5, rows).clip(69, 87).round(1)
    weight = (height * 2.5 + rng.normal(5, 18, rows)).clip(165, 275).round()
    games = rng.integers(22, 36, rows)
    minutes = games * rng.uniform(20, 34, rows)
    usage = rng.uniform(.15, .31, rows)
    fga = minutes * usage / 2.05
    three_rate = rng.beta(2.3, 3.1, rows).clip(.03, .8)
    three_pa = fga * three_rate
    three_pct = rng.normal(.345, .055, rows).clip(.2, .48)
    three_p = three_pa * three_pct
    fta = fga * rng.uniform(.18, .52, rows)
    ft_pct = rng.normal(.745, .075, rows).clip(.5, .92)
    ft = fta * ft_pct
    two_makes = (fga - three_pa) * rng.normal(.515, .06, rows).clip(.35, .68)
    pts = 3 * three_p + 2 * two_makes + ft
    ast = minutes * rng.uniform(.025, .19, rows)
    tov = np.maximum(8, ast / rng.uniform(.9, 3.2, rows))
    orb = minutes * rng.uniform(.008, .085, rows)
    drb = minutes * rng.uniform(.045, .19, rows)
    stl = minutes * rng.uniform(.012, .045, rows)
    blk = minutes * rng.uniform(.002, .075, rows) * (height / 78) ** 3

    college = pd.DataFrame({
        "player": [f"Demo Prospect {i + 1:03d}" for i in idx],
        "season": 2025, "age": age, "height_in": height, "weight_lb": weight,
        "games": games, "minutes": minutes.round(0), "pts": pts.round(0),
        "fga": fga.round(0), "fta": fta.round(0), "three_pa": three_pa.round(0),
        "three_p": three_p.round(0), "ft": ft.round(0), "ast": ast.round(0),
        "tov": tov.round(0), "orb": orb.round(0), "drb": drb.round(0),
        "stl": stl.round(0), "blk": blk.round(0),
        "school": [f"Demo University {(i % 20) + 1}" for i in idx],
        "conference": [f"Conference {chr(65 + i % 5)}" for i in idx],
        "draft_pick": np.where(idx < 60, idx + 1, np.nan),
        "class_year": np.select([age < 19.5, age < 20.5, age < 21.5], ["FR", "SO", "JR"], default="SR"),
    })

    # Synthetic outcome signal deliberately combines age, efficiency, playmaking,
    # size, and noise so the demo tests the full modeling workflow realistically.
    ts = pts / (2 * (fga + .44 * fta))
    signal = (ts - .5) * 24 + (ast / np.maximum(tov, 1)) * .8 + (23 - age) * .35 + (height - 76) * .08
    bpm = signal + rng.normal(0, 1.7, rows) - 3
    nba_minutes = np.maximum(0, 350 + signal * 260 + rng.normal(0, 500, rows)).round()
    nba_ws = np.maximum(-1, signal * 1.15 + rng.normal(0, 1.8, rows)).round(2)
    outcomes = pd.DataFrame({"player": college.player, "nba_minutes": nba_minutes, "nba_ws": nba_ws, "nba_bpm": bpm.round(2)})

    RAW.mkdir(parents=True, exist_ok=True)
    college.to_csv(RAW / "college_players.csv", index=False)
    outcomes.to_csv(RAW / "nba_outcomes.csv", index=False)
    print(f"Generated {rows} synthetic prospect histories in {RAW}")


if __name__ == "__main__":
    main()

