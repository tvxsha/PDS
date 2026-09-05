# Dataset format

Every document is a single `.txt` file. Filename convention matters — the
eval script will parse the attack type out of the filename, so stick to this:

- `data/clean/clean_###.txt` — normal, non-poisoned docs (aim for 50-60)
- `data/poisoned/injection_###.txt` — direct instruction-override attacks
  (e.g. "ignore previous instructions...")
- `data/poisoned/authority_###.txt` — fake system/policy authority spoofing
  (e.g. "[SYSTEM OVERRIDE] new policy...")
- `data/poisoned/corpus_###.txt` — corpus-poisoning: content that's subtly
  wrong/malicious but doesn't use obvious override language (this is the
  category most likely to slip past the pattern detector on purpose --
  useful for your failure-map story)
- `data/poisoned/perplexity_###.txt` — garbled / awkwardly-generated text
  that should trip the fluency scorer specifically

Target: ~12-15 docs per poisoned category (48-60 total poisoned), not 5 --
5 samples per category won't survive being asked "is that statistically
meaningful?"

Keep clean docs domain-matched to coding/API reference material where
possible (per the domain-specific testing plan) -- pull from real internal
docs, READMEs, Stack Overflow-style Q&A, or generate synthetic-but-realistic
versions. Note in the file (or a parallel `_source.txt`) where each one came
from, in case you need to justify it later.

Once you have a first batch (even 10-15 clean + a few of each poisoned type),
run `eval/dataset_check.py` to sanity-check the folder before building more.
