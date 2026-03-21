from __future__ import annotations

import math


def normalize_seed_loss(seed: int, raw: object) -> dict[str, object]:
    if isinstance(raw, (int, float)):
        return {
            "seed": int(seed),
            "total_loss": float(raw),
            "components": {},
        }

    if isinstance(raw, dict):
        if "total_loss" in raw:
            components = {
                str(k): float(v)
                for k, v in dict(raw.get("components", {})).items()
            }
            out = {
                "seed": int(seed),
                "total_loss": float(raw["total_loss"]),
                "components": components,
            }
            for k, v in raw.items():
                if k not in {"seed", "total_loss", "components"}:
                    out[k] = v
            return out

        components = {str(k): float(v) for k, v in raw.items()}
        values = list(components.values())
        total = float(sum(values) / len(values)) if values else math.nan
        return {
            "seed": int(seed),
            "total_loss": total,
            "components": components,
        }

    if isinstance(raw, list):
        rows = [normalize_seed_loss(seed, item) for item in raw]
        totals = [float(row["total_loss"]) for row in rows]
        total = float(sum(totals) / len(totals)) if totals else math.nan
        return {
            "seed": int(seed),
            "total_loss": total,
            "components": {},
            "items": totals,
            "n_items": len(totals),
        }

    raise TypeError("seed_loss_fn must return float, dict, or list")


def summarize_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("rows must not be empty")

    totals = [float(row["total_loss"]) for row in rows]
    mean_total = sum(totals) / len(totals)
    var = sum((x - mean_total) ** 2 for x in totals) / len(totals)
    std = var ** 0.5

    return {
        "n_seeds": len(rows),
        "mean_total_loss": float(mean_total),
        "std_total_loss": float(std),
        "stderr_total_loss": float(std / (len(rows) ** 0.5)),
        "total_losses": totals,
    }
