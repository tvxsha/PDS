"""
Shared score plumbing (roadmap task 1.9).

What lives here
  * per-document signal caches   eval/results/signal_<name>_seed<N>_<splithash>.csv   (id,score,ms)
  * MiniLM embedding cache       eval/results/emb_minilm_seed<N>_<splithash>.npz      (git-ignored)
  * evaluation sets              data/evalsets/<name>.jsonl  (Row schema, never part of the pool)
  * scores of an evaluation set  eval/results/external_<set>_seed<N>_<splithash>.csv  (id,signal,score,ms)

Rules built into the code
  * The held-out role is refused unless final=True. Only eval/final_test.py passes final=True.
  * Caches are append-only and resumable: a crash, a laptop sleep or Ctrl+C loses at most 25 documents.
  * No torch / transformers import at module import time (CI has neither); detectors are
    imported inside the functions that need them.
  * A cache file may contain rows of roles you did not ask for (after the final run it contains
    the held-out ids). Always pick documents through the roles you request.
"""

import csv
import hashlib
import json
import math
import os
import re
import time

import numpy as np

from eval.protocol import POOLED_PATH, ROLES, load_split, read_jsonl, split_hash

SIGNAL_DIR = "eval/results"
EVALSET_DIR = "data/evalsets"
BASE_SIGNALS = ("pattern", "embedding", "perplexity")
FLUSH_EVERY = 25
_NAME_OK = re.compile(r"[A-Za-z0-9_.\-]+")


# --------------------------------------------------------------------------
# small helpers (tests replace _split_hash and _rows_for_roles)
# --------------------------------------------------------------------------

def _check_name(name):
    if not _NAME_OK.fullmatch(name or ""):
        raise ValueError(f"bad name {name!r}: use letters, digits, '_', '-', '.' only")


def _check_roles(roles, final):
    bad = [r for r in roles if r not in ROLES]
    if bad:
        raise ValueError(f"unknown role(s) {bad}; valid roles: {list(ROLES)}")
    if "test" in roles and not final:
        raise ValueError("test role requested (final=True is for eval/final_test.py only)")


def _split_hash(seed):
    return split_hash(load_split(seed))


def _rows_for_roles(seed, roles):
    """Rows of the requested roles only, in split-file order. Other roles are never turned into rows."""
    split = load_split(seed)
    by_id = {r.id: r for r in read_jsonl(POOLED_PATH)}
    return [by_id[i] for role in roles for i in split["roles"][role]]


def _repair(path):
    """Drop a half-written last line (crash or sleep during a flush) so appending starts on a clean line."""
    if not os.path.exists(path):
        return
    with open(path, "rb") as f:
        data = f.read()
    if data and not data.endswith(b"\n"):
        with open(path, "wb") as f:
            f.write(data[:data.rfind(b"\n") + 1])


def _read_table(path):
    """All complete lines of a CSV as lists of fields (header included). A half-written last line is ignored."""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8", newline="") as f:
        text = f.read()
    lines = text.split("\n")[:-1]          # what follows the last newline is empty or half-written
    return list(csv.reader(lines))


def _is_new(path):
    return (not os.path.exists(path)) or os.path.getsize(path) == 0


def _progress(label, k, n, t_start):
    el = time.perf_counter() - t_start
    print(f"  [{label}] {k}/{n} done, {el / 60:.1f} min elapsed, about {el / k * (n - k) / 60:.1f} min left", flush=True)


# --------------------------------------------------------------------------
# per-document signals
# --------------------------------------------------------------------------

def signal_path(name, seed):
    _check_name(name)
    return os.path.join(SIGNAL_DIR, f"signal_{name}_seed{seed}_{_split_hash(seed)}.csv")


def load_signal(name, seed):
    """{id: (score, ms)}; {} if the signal has not been computed yet."""
    out = {}
    for rec in _read_table(signal_path(name, seed))[1:]:
        try:
            out[rec[0]] = (float(rec[1]), float(rec[2]))
        except (IndexError, ValueError):
            continue
    return out


def score_missing(name, seed, roles, scorer, final=False):
    """Score the documents of `roles` that this signal has not seen yet; return {id: (score, ms)}.

    scorer(row) -> float, where row is an eval.protocol.Row. Appends "id,score,ms" lines, flushing
    every 25 documents. ms is wall time of the scorer call (time.perf_counter). The returned dict
    holds everything in the cache file (possibly more ids than the roles you asked for)."""
    roles = list(dict.fromkeys(roles))
    _check_roles(roles, final)
    path = signal_path(name, seed)
    cache = load_signal(name, seed)
    todo = [r for r in _rows_for_roles(seed, roles) if r.id not in cache]
    if todo:
        print(f"[{name}] seed {seed}: {len(todo)} documents to score, {len(cache)} cached ({path}). "
              "Ctrl+C is safe: run the same command again to resume.", flush=True)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        _repair(path)
        new_file = _is_new(path)
        t_start = time.perf_counter()
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            if new_file:
                w.writerow(["id", "score", "ms"])
            for k, row in enumerate(todo, 1):
                t = time.perf_counter()
                score = float(scorer(row))
                ms = (time.perf_counter() - t) * 1000.0
                if not math.isfinite(score):
                    raise ValueError(f"scorer returned {score} for {row.id}")
                w.writerow([row.id, repr(score), f"{ms:.3f}"])
                cache[row.id] = (score, ms)
                if k % FLUSH_EVERY == 0 or k == len(todo):
                    f.flush()
                    _progress(name, k, len(todo), t_start)
    return cache


# --------------------------------------------------------------------------
# MiniLM embeddings
# --------------------------------------------------------------------------

def embeddings_path(seed):
    return os.path.join(SIGNAL_DIR, f"emb_minilm_seed{seed}_{_split_hash(seed)}.npz")


def _encode(texts):
    from detectors.embedding_detector import _get_model       # loads sentence-transformers only now
    model = _get_model()
    return np.asarray(model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False),
                      dtype=np.float32)


def get_embeddings(seed, roles, final=False):
    """(ids, matrix): L2-normalised all-MiniLM-L6-v2 embeddings (float32, one row per id) of the documents
    of `roles`, in split-file order. Cached per id in emb_minilm_seed<N>_<splithash>.npz and extended when
    new ids are requested. Encoded in batches, so values can differ from one-at-a-time encoding by ~1e-7."""
    roles = list(dict.fromkeys(roles))
    _check_roles(roles, final)
    rows = _rows_for_roles(seed, roles)
    path = embeddings_path(seed)
    ids_c, mat_c = [], None
    if os.path.exists(path):
        with np.load(path, allow_pickle=False) as z:
            ids_c, mat_c = [str(i) for i in z["ids"]], z["emb"]
    have = set(ids_c)
    todo = [r for r in rows if r.id not in have]
    if todo:
        print(f"[embeddings] seed {seed}: encoding {len(todo)} documents ({len(ids_c)} cached)", flush=True)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        t_start = time.perf_counter()
        for a in range(0, len(todo), 256):
            chunk = todo[a:a + 256]
            new = _encode([r.text for r in chunk])
            mat_c = new if mat_c is None else np.vstack([mat_c, new])
            ids_c += [r.id for r in chunk]
            tmp = path[:-4] + ".tmp.npz"
            np.savez(tmp, ids=np.array(ids_c), emb=mat_c)
            os.replace(tmp, path)                                # a crash never leaves a broken file
            _progress("embeddings", min(a + 256, len(todo)), len(todo), t_start)
    index = {i: k for k, i in enumerate(ids_c)}
    ids = [r.id for r in rows]
    if not ids:
        return ids, np.zeros((0, 384), dtype=np.float32)
    return ids, mat_c[[index[i] for i in ids]]


# --------------------------------------------------------------------------
# evaluation sets (data/evalsets/*.jsonl) and their scores
# --------------------------------------------------------------------------

def evalset_path(set_name):
    _check_name(set_name)
    return os.path.join(EVALSET_DIR, f"{set_name}.jsonl")


def load_evalset(set_name):
    """list[Row] of data/evalsets/<set_name>.jsonl. Checks the conventions of Section 3 of ROADMAP.md."""
    path = evalset_path(set_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Evaluation sets are built by the scripts in data/evalsets/.")
    rows = read_jsonl(path)
    bad = sorted({r.source for r in rows if not r.source.startswith("ext_")})
    if bad:
        raise ValueError(f"{path}: source must start with 'ext_' (found {bad}); evaluation sets are never pool sources")
    ids = [r.id for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate ids")
    return rows


def external_path(set_name, seed):
    _check_name(set_name)
    return os.path.join(SIGNAL_DIR, f"external_{set_name}_seed{seed}_{_split_hash(seed)}.csv")


def load_external(set_name, seed):
    """{signal: {id: (score, ms)}} from the external cache; {} if it does not exist."""
    out = {}
    for rec in _read_table(external_path(set_name, seed))[1:]:
        try:
            out.setdefault(rec[1], {})[rec[0]] = (float(rec[2]), float(rec[3]))
        except (IndexError, ValueError):
            continue
    return out


def _base_scorers(seed, names):
    """{signal: fn(text) -> float} for the base detectors in `names`. Models are loaded here, on demand.
    The embedding baseline is built from THIS seed's baseline-role documents (as protocol_eval does)."""
    out = {}
    if "pattern" in names:
        from detectors import pattern_detector
        out["pattern"] = lambda t: pattern_detector.score_document(t).score
    if "embedding" in names:
        from detectors import embedding_detector
        baseline = embedding_detector.BaselineCorpus([r.text for r in _rows_for_roles(seed, ["baseline"])])
        out["embedding"] = lambda t: embedding_detector.score_document(t, baseline).score
    if "perplexity" in names:
        from detectors import perplexity_detector
        perplexity_detector._get_model_and_tokenizer()
        out["perplexity"] = lambda t: perplexity_detector.score_document(t).score
    return out


def score_external(set_name, seed, extra=None):
    """Score every document of an evaluation set with the three base detectors (signals "pattern",
    "embedding", "perplexity"; embedding baseline = this seed's baseline documents) plus any extra
    {signal_name: fn(text) -> float}. Cached in external_path(); returns {signal: {id: score}}.
    If the evaluation set file changed since it was scored, everything is scored again."""
    extra = dict(extra or {})
    for s in extra:
        _check_name(s)
    if set(extra) & set(BASE_SIGNALS):
        raise ValueError(f"extra signal names must differ from {list(BASE_SIGNALS)}")
    rows = load_evalset(set_name)
    path = external_path(set_name, seed)
    stamp = path + ".src"                                        # fingerprint of the ids + texts that were scored
    sha = hashlib.sha1(json.dumps([[r.id, r.text] for r in rows]).encode("utf-8")).hexdigest()
    old = None
    if os.path.exists(stamp):
        with open(stamp, encoding="utf-8") as f:
            old = f.read().strip()
    if os.path.exists(path) and old != sha:
        print(f"[{set_name}] the evaluation set changed since it was scored: scoring everything again", flush=True)
        os.remove(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(stamp, "w", encoding="utf-8") as f:
        f.write(sha + "\n")

    cache = load_external(set_name, seed)
    signals = list(BASE_SIGNALS) + list(extra)
    missing = {s: [r for r in rows if r.id not in cache.get(s, {})] for s in signals}
    need = [s for s in BASE_SIGNALS if missing[s]]
    scorers = dict(_base_scorers(seed, need)) if need else {}
    scorers.update(extra)
    if any(missing.values()):
        _repair(path)
        new_file = _is_new(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            if new_file:
                w.writerow(["id", "signal", "score", "ms"])
            for s in signals:
                if not missing[s]:
                    continue
                print(f"[{set_name}] seed {seed}: scoring {len(missing[s])} documents with '{s}'", flush=True)
                for r in missing[s]:
                    t = time.perf_counter()
                    score = float(scorers[s](r.text))
                    ms = (time.perf_counter() - t) * 1000.0
                    if not math.isfinite(score):
                        raise ValueError(f"signal {s} returned {score} for {r.id}")
                    w.writerow([r.id, s, repr(score), f"{ms:.3f}"])
                    cache.setdefault(s, {})[r.id] = (score, ms)
                f.flush()
    return {s: {r.id: cache[s][r.id][0] for r in rows} for s in signals}