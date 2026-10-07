"""
Length-matched clean sampling (roadmap task 2.3).

If clean documents are systematically longer or shorter than poisoned ones,
a detector can score well just by measuring length. So whenever a source has
poisoned docs but no natural clean counterpart (PoisonedRAG), the clean docs
are chosen to match the poisoned docs' length distribution one-for-one.
"""

import random
import statistics
from bisect import bisect_left


def match_by_length(candidates, target_lengths, seed=42):
    """
    candidates:     list of (key, text) clean documents to choose from
    target_lengths: list of ints, one per poisoned doc to be matched

    For each target length (visited in random order) take the not-yet-used
    candidate whose length is closest. Returns a list of (key, text), at
    most len(target_lengths) long (fewer if candidates run out).
    """
    pool = sorted(candidates, key=lambda c: (len(c[1]), str(c[0])))
    lens = [len(c[1]) for c in pool]
    used = [False] * len(pool)
    targets = list(target_lengths)
    random.Random(seed).shuffle(targets)

    picks = []
    for t in targets:
        i = bisect_left(lens, t)
        lo, hi = i - 1, i
        chosen = None
        while lo >= 0 or hi < len(pool):
            while lo >= 0 and used[lo]:
                lo -= 1
            while hi < len(pool) and used[hi]:
                hi += 1
            if lo < 0 and hi >= len(pool):
                break
            if lo < 0:
                chosen = hi
            elif hi >= len(pool):
                chosen = lo
            else:
                chosen = lo if (t - lens[lo]) <= (lens[hi] - t) else hi
            break
        if chosen is None:
            break
        used[chosen] = True
        picks.append(pool[chosen])
    return picks


def length_summary(rows):
    """(n, median, p25, p75) of document length in characters."""
    if not rows:
        return (0, 0, 0, 0)
    ls = sorted(r.length for r in rows)
    q = statistics.quantiles(ls, n=4) if len(ls) >= 2 else [ls[0]] * 3
    return (len(ls), int(statistics.median(ls)), int(q[0]), int(q[2]))