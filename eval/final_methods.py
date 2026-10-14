"""
Registries for the one-shot final evaluation (roadmap task 1.7a).

eval/final_test.py runs every method named in eval/final_config.json. A method is a function
registered here; other cards add their entries to THIS file and nothing else needs editing
(apart from the "methods" / "extras" lists in eval/final_config.json):

    @register_method("my method")
    def build(ctx):
        # return (train_scores, eval_scores): one float per document of ctx.train and ctx.eval, in that order.
        # train_scores must be OUT-OF-FOLD for anything that is fitted (the threshold comes from the clean ones).
        ...

    @register_extra("my extra")
    def run(ctx):
        # results that are not one score per document (for example the cascade of task 5.4).
        # return {"file_key": DataFrame}; each frame is written to final_<mode>_seed<N>_<file_key>.csv
        ...

`ctx` is a Context (below). Documents are dicts as returned by eval.sentence_eval.load_docs:
id, text, y (0/1), src, grp, pat, emb, ppl (and attack, the attack type, added by final_test).
ctx.config is eval/final_config.json as a dict, so a card that needs a setting (embedding variant, cascade
settings) reads it there and adds the key to the JSON file.
ctx.eval holds the VALIDATION documents in a dry run and the held-out documents in the final run; the code that
builds a method is identical in both cases.

No torch import at module import time.
"""

import numpy as np

from eval.apply_models import METHODS as BASE_METHODS

METHODS = {}   # name -> build(ctx) -> (train_scores, eval_scores)
EXTRAS = {}    # name -> run(ctx) -> {file_key: DataFrame}


def register_method(name):
    def deco(fn):
        if name in METHODS:
            raise ValueError(f"method {name!r} is registered twice")
        METHODS[name] = fn
        return fn
    return deco


def register_extra(name):
    def deco(fn):
        if name in EXTRAS:
            raise ValueError(f"extra {name!r} is registered twice")
        EXTRAS[name] = fn
        return fn
    return deco


class Context:
    def __init__(self, seed, eval_role, final, train, eval_docs, scores_path, split_hash, config=None):
        self.seed = seed
        self.config = config or {}          # eval/final_config.json as a dict (extra keys added by other cards)
        self.eval_role = eval_role          # "val" (dry run) or the held-out role (final run)
        self.final = final                  # True only in the final run
        self.train = train
        self.eval = eval_docs
        self.scores_path = scores_path
        self.split_hash = split_hash
        self._fitted = None
        self._eval_base = None
        self._published = None

    @property
    def y_train(self):
        return np.array([d["y"] for d in self.train])

    @property
    def y_eval(self):
        return np.array([d["y"] for d in self.eval])

    def fitted(self):
        """eval.apply_models.FittedModels for this seed (fitted once, on TRAIN only)."""
        if self._fitted is None:
            from eval.apply_models import fit_models
            self._fitted = fit_models(self.seed, self.scores_path)
        return self._fitted

    def base_scores(self):
        """{base method name: scores of the eval documents}."""
        if self._eval_base is None:
            self._eval_base = self.fitted().score(self.eval)
        return self._eval_base

    def signal(self, name, scorer):
        """A cached per-document signal (eval.signal_cache.score_missing) as arrays aligned with
        ctx.train and ctx.eval: returns (train_array, eval_array). scorer(row) -> float, row is a Row."""
        from eval import signal_cache as sc
        cache = sc.score_missing(name, self.seed, ["train", self.eval_role], scorer, final=self.final)
        return (np.array([cache[d["id"]][0] for d in self.train]),
                np.array([cache[d["id"]][0] for d in self.eval]))

    def published(self):
        """Margins of the published classifier for ctx.train and ctx.eval: (train_array, eval_array).
        Documents missing from the shared cache are scored now (the held-out documents in a final run)."""
        if self._published is None:
            from detectors import published_classifier as pc
            from eval import baseline_eval as be
            path = be.cache_path(pc.DEFAULT_MODEL, pc.MAX_WINDOWS)
            cache, _ = be.score_missing(self.train + self.eval, pc.DEFAULT_MODEL, pc.MAX_WINDOWS, path)
            self._published = (np.array([cache[d["id"]]["margin"] for d in self.train]),
                               np.array([cache[d["id"]]["margin"] for d in self.eval]))
        return self._published


# ---- the methods of eval.fpr_eval.fit_everything (fitted by eval.apply_models.fit_models) ----------

def _register_base(name):
    @register_method(name)
    def build(ctx):
        return ctx.fitted().train_scores[name], ctx.base_scores()[name]
    return build


for _name in BASE_METHODS:
    _register_base(_name)


# ---- the published classifier (not fitted on our data: the train split only fixes the cut-off) ------

@register_method("published classifier")
def _published(ctx):
    return ctx.published()


# ---- the feature set chosen by task 4.10 (eval/results/fusion_v2_candidate.json) -------------------------

@register_method("fusion candidate")
def _fusion_candidate(ctx):
    from eval import fusion_v2
    return fusion_v2.candidate_scores(ctx)


# ---- the cost-aware cascade of task 5.4 (an extra: needs eval/results/latency_benchmark.json from task 1.3) ------

@register_extra("cascade")
def _cascade(ctx):
    from eval import cascade_eval
    return cascade_eval.cascade_extra(ctx)