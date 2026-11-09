"""Write the analytical rule comparison. Run from the jw directory."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.compare import heterogeneous_table

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    table = heterogeneous_table()
    out = ROOT / "results" / "offline_rules.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    for rule, cost in table["mean_realized_cost"].items():
        print(f"{rule:16} mean realized extra cost {cost:.1f}")


if __name__ == "__main__":
    main()
