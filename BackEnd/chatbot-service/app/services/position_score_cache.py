_cache: dict[int, float] = {}


def get(position_id: int, default: float = 70.0) -> float:
    return _cache.get(position_id, default)


def set_score(position_id: int, score: float) -> None:
    _cache[position_id] = score


def preload(scores: dict) -> None:
    """Bulk-load positionId → score map fetched from recruitment-service on startup."""
    _cache.update({int(k): float(v) for k, v in scores.items()})
