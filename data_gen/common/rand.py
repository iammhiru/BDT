from __future__ import annotations
import random, xxhash

def seeded_rng(*parts: str) -> random.Random:
    seed = xxhash.xxh64_intdigest("|".join(parts))
    return random.Random(seed)
