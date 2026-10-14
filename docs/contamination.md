# Pretrained-model contamination check

**Checked:** 2026-10-14. This is a documentation review, not a forensic test of model weights. A missing disclosure is recorded as *not documented*, not as evidence that a model did not see a dataset.

## Models

| Model | Documented training data (short quotation) | Source | What can and cannot be concluded |
|---|---|---|---|
| `gpt2` (GPT-2) | “they scraped all the web pages from outbound links on Reddit which received at least 3 karma.” | [GPT-2 model card, Training data](https://huggingface.co/openai-community/gpt2) | The card says the underlying WebText corpus was not publicly released. It also says Wikipedia pages were removed. It does not provide a document-level membership check for this project’s data or establish whether all near-duplicate text was absent. |
| `sentence-transformers/all-MiniLM-L6-v2` | “We use the concatenation from multiple datasets to fine-tune the model.” | [MiniLM model card, Training procedure](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | The card lists large sources including Reddit comments, S2ORC citation pairs, WikiAnswers, PAQ, and Stack Exchange. The listed sources are broad enough that overlap or near-duplicates cannot be ruled out from documentation alone. The card does not document membership of PDS evaluation documents. |
| `protectai/deberta-v3-base-prompt-injection-v2` | “The dataset used for training the model was meticulously assembled from various public open datasets” | [Protect AI model card, Dataset](https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2) | The card also describes crafted prompt injections based on research, articles, security competitions, and community feedback, and lists dataset licenses. It does not provide a complete item-level training manifest that would let us verify whether PDS documents or close variants were included. This is the most directly task-related model, so overlap risk is especially important to acknowledge. |

## Datasets represented in the pool

| Dataset | Release information checked | Licence information checked | Relevance to contamination |
|---|---|---|---|
| BIPIA | The official repository and benchmark are publicly available; the associated paper first appeared as arXiv:2312.14197 in December 2023. The repository’s exact initial dataset publication date is **not documented here**. | The repository says its code licence is MIT, but explicitly notes that benchmark components have their own licences. It lists WikiTableQuestions as CC BY-SA 4.0, Stack Exchange as CC BY-SA 4.0, and OpenAI Evals invoice data as MIT; some tasks require users to obtain source data under its original terms. | GPT-2 predates BIPIA, but chronology alone does not prove non-overlap. MiniLM and Protect AI may have been trained or fine-tuned later; their cards do not provide a full item-level comparison with this checkout. |
| PoisonedRAG | The paper appeared as arXiv:2402.07867 in February 2024; the authors’ repository reports acceptance to USENIX Security 2025 on 20 June 2024. Exact first release date of every dataset artifact is **not documented here**. | The authors’ repository is marked MIT. That repository licence should not be treated as a blanket licence for every upstream dataset used by the project. The underlying Natural Questions/BEIR material retains its source terms. | The Protect AI card mentions prompt injections assembled from public data and research/community sources, but does not name PoisonedRAG. Direct inclusion is therefore **not documented**. |
| BEIR Natural Questions (NQ) passages | Google’s official Natural Questions page describes the dataset and download; the exact release date of the specific BEIR-preprocessed NQ snapshot used in this project is **not documented here**. | Google’s Natural Questions page states that NQ is released under **Creative Commons Share-Alike 3.0**. Check the exact snapshot and its source terms before redistribution. | GPT-2’s card says Wikipedia was removed from WebText, which lowers one obvious overlap route for Wikipedia-derived passages, but does not prove that no text overlaps through other sources. The MiniLM card lists Google Natural Questions among datasets used for model training; this is a documented potential overlap risk for the NQ-derived clean corpus. |

## Conclusion

The documentation does not establish that any evaluated model was trained on the exact PDS evaluation records. It also does not justify a blanket claim that the models are uncontaminated. MiniLM’s model card explicitly lists Natural Questions among its training datasets, making overlap with the BEIR NQ-derived passages a documented risk at the dataset-family level. Protect AI’s classifier was trained on prompt-injection data, but its card does not provide an item-level manifest. GPT-2’s card says Wikipedia was removed from WebText, which is useful but not a full duplicate check. These findings are disclosure-based, not proof of memorization or non-memorization.

## Official sources checked

- GPT-2 model card: <https://huggingface.co/openai-community/gpt2>
- MiniLM model card: <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>
- Protect AI model card: <https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2>
- BIPIA repository and component licences: <https://github.com/microsoft/BIPIA> and <https://github.com/microsoft/BIPIA/blob/main/LICENSE>
- PoisonedRAG authors’ repository: <https://github.com/sleeepeer/PoisonedRAG>
- Natural Questions official data page: <https://ai.google.com/research/NaturalQuestions/download>
- BIPIA paper: <https://arxiv.org/abs/2312.14197>
- PoisonedRAG paper: <https://arxiv.org/abs/2402.07867>
