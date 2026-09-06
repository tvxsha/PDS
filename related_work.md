# Related Work

This section reviews recent work (2023–2025) relevant to the three signals used in the Poison Detection System (PDS): embedding-based anomaly detection, language-model fluency/perplexity, and instruction/prompt-pattern detection.

## 1. Benchmarking and Defending Against Indirect Prompt Injection Attacks on Large Language Models (Yi et al., 2023)

**Source:** Jingwei Yi et al., *Benchmarking and Defending Against Indirect Prompt Injection Attacks on Large Language Models*, 2023. https://arxiv.org/abs/2312.14197

**Summary:** The paper introduces BIPIA, a benchmark for indirect prompt injection, where malicious instructions are hidden inside external content that an LLM later processes. The authors show that LLMs are broadly vulnerable because they struggle to distinguish ordinary retrieved information from instructions that should actually be followed.

**Connection to PDS:** This directly motivates the instruction-pattern detector. PDS uses explicit indicators such as “ignore previous instructions” and fake system-style directives as a lightweight first-pass signal for the same class of malicious content.

## 2. Defending Against Indirect Prompt Injection Attacks With Spotlighting (Hines et al., 2024)

**Source:** Keegan Hines et al., *Defending Against Indirect Prompt Injection Attacks With Spotlighting*, 2024. https://arxiv.org/abs/2403.14720

**Summary:** Spotlighting addresses the problem of mixing trusted instructions with untrusted external text by transforming the input so the model can better distinguish different sources of information. Their experiments show that making provenance more explicit can substantially reduce the success of indirect prompt injection attacks.

**Connection to PDS:** The work supports the idea that malicious instructions embedded in otherwise normal documents are a meaningful security signal. It complements our pattern detector concept, while also showing why simple keyword matching is only one layer of defense rather than a complete solution.

## 3. Towards More Robust Retrieval-Augmented Generation: Evaluating RAG Under Adversarial Poisoning Attacks (Su et al., 2024)

**Source:** Jinyan Su et al., *Towards More Robust Retrieval-Augmented Generation: Evaluating RAG Under Adversarial Poisoning Attacks*, 2024. https://arxiv.org/abs/2412.16708

**Summary:** This study evaluates how adversarially poisoned passages affect both retrieval and generation in RAG systems. It examines why poisoned passages are retrieved and also explores whether skeptical prompting can reduce their influence on generated answers.

**Connection to PDS:** This is closely aligned with our overall threat model: a malicious document can first win retrieval and then influence generation. It supports evaluating detection at the document/retrieval level rather than assuming that the final LLM response will always expose the attack.

## 4. CED: Comparing Embedding Differences for Detecting Out-of-Distribution and Hallucinated Text (Lee et al., 2024)

**Source:** Hakyung Lee et al., *CED: Comparing Embedding Differences for Detecting Out-of-Distribution and Hallucinated Text*, Findings of EMNLP 2024. https://aclanthology.org/2024.findings-emnlp.874/

**Summary:** CED proposes a training-free approach for distinguishing in-distribution and out-of-distribution text using differences in embedding representations. The paper highlights a key difficulty for text anomaly detection: normal and anomalous examples can occupy overlapping regions of embedding space, so the choice of representation and comparison method matters.

**Connection to PDS:** This provides direct support for our embedding-anomaly detector. PDS similarly represents documents numerically and flags documents that are unusually distant or inconsistent with the normal reference corpus, while the paper's discussion of embedding overlap is relevant to our planned blind-spot analysis.

## 5. Can Indirect Prompt Injection Attacks Be Detected and Removed? (Chen et al., 2025)

**Source:** Yulin Chen et al., *Can Indirect Prompt Injection Attacks Be Detected and Removed?*, ACL 2025. https://aclanthology.org/2025.acl-long.890/

**Summary:** This work focuses specifically on detecting and removing indirect prompt injections coming from external sources such as search results. The authors evaluate existing LLMs and open-source detectors and also investigate methods that identify and remove injected instructions from documents.

**Connection to PDS:** The paper reinforces the importance of treating retrieved documents as potentially untrusted input. Its focus on identifying injected instruction spans connects directly to our pattern-based detector, while its benchmark perspective is useful when designing and interpreting our cross-domain test set.

## 6. Practical Poisoning Attacks against Retrieval-Augmented Generation (Zhang et al., 2025)

**Source:** Baolei Zhang et al., *Practical Poisoning Attacks against Retrieval-Augmented Generation*, 2025. https://arxiv.org/abs/2504.03957

**Summary:** The paper introduces CorruptRAG, a poisoning attack designed to work by injecting only a single malicious text into a RAG knowledge base rather than relying on many poisoned documents. The result emphasizes that poisoning can be both practical and stealthy, even when the attacker has limited ability to modify the corpus.

**Connection to PDS:** This is important for our evaluation because a detector should not assume that poisoning is obvious from volume or repetition. A single carefully written malicious document may need to be identified using multiple complementary signals, making the combination of embedding anomaly, fluency/perplexity, and instruction-pattern scores particularly relevant.

## Overall relevance to PDS

The literature supports the three components of our system but also reinforces the project's intended honesty about novelty. Prior work already establishes that RAG poisoning is a real threat, that indirect instructions can be hidden in external content, and that embeddings and language-model scores can provide useful anomaly signals. Our contribution is therefore not claiming a new detection algorithm; instead, the project compares these existing signals, measures their latency and individual effectiveness, and studies whether their failure modes complement one another.

This framing is consistent with the project pitch, which explicitly states that the three detection techniques are drawn from published research and that the intended novelty is comparative failure/blind-spot and latency analysis rather than inventing a new detector.
