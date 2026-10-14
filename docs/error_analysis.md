# Manual detector error analysis

> This is a reporting template, not a result. Generate `eval/results/error_dump_seed42.csv`, inspect every row, fill `cause` and `notes`, save as `eval/results/error_dump_seed42_annotated.csv`, and only then complete this summary. Do not invent counts or examples.

## Protocol

- Split: validation only, seed 42.
- Method: `LR + sentence classifier`.
- Threshold: train-derived 5% clean false-alarm budget.
- False positives: include every validation false positive.
- False negatives: seeded sample of up to 60, stratified proportionally by source.

## Causes observed

| Error type | Count | Source breakdown | What could help detect it? |
|---|---:|---|---|
| no-instruction-words | TODO | TODO | TODO |
| paraphrase | TODO | TODO | TODO |
| buried-in-long-doc | TODO | TODO | TODO |
| code-context | TODO | TODO | TODO |
| benign-imperative | TODO | TODO | TODO |
| template-text | TODO | TODO | TODO |
| unusual-domain | TODO | TODO | TODO |
| other | TODO | TODO | TODO |

## Short examples

Add up to three short verbatim examples per observed cause, drawn from the reviewed CSV. Do not include more text than needed to illustrate the failure mode.

## Limitations

This is a qualitative review of validation errors, not an unbiased estimate of deployment error prevalence. False negatives are sampled, so report both the total number available and the number reviewed. No test data were used.
