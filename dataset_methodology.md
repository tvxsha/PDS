✅ Clean (data/clean/) — matches, no issues

55/55 files accounted for in clean_sources.tsv
All sourced from 10 real open-source READMEs (axios, django, express, fastapi, flask, nodejs, numpy, pandas, redis, requests), method = real_readme_paragraph throughout
Minor note: distribution is uneven — flask=1, django=2, but axios/fastapi/nodejs/pandas/redis=8 each. Not an error, just worth one line in the methodology write-up.

✅ Injection + Authority (data/poisoned/injection_*, authority_*) — matches

30/30 files (15+15) accounted for in injection_authority_sources.tsv
Consistent with generate_injection_authority.py: template-generated with randomized phrasing, source labeled generated

✅ Perplexity (data/poisoned/perplexity_*) — matches

15/15 files accounted for in perplexity_sources.tsv
Consistent with generate_perplexity_docs.py: real sentences corrupted via word-order shuffling, source labeled generated_from_base_sentence

🚩 Corpus (data/poisoned/corpus_*) — real gap, needs flagging to Tvisha

All 15 corpus_*.txt files exist and are actively used everywhere in the eval code (run_evaluation.py, calibrated_fusion.py, train_learned_fusion*.py, and it's the category the embedding detector's docstring specifically calls out)
But there is no provenance file or entries anywhere for this category — no corpus_sources.tsv, and no generate_corpus.py-style script the way the other two categories have. I can't tell from the repo whether these were hand-written, template-generated, or adapted from something else.
This matters more than it might look: corpus-poisoning is the category the roadmap flags as central to the failure-map story, so if a reviewer asks "how was this constructed," there's currently no answer on record.
