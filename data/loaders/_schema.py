"""
Shared row schema for all data/loaders/*.py modules (task 2.2, also used by
task 2.1's loaders).

NOTE: this is NOT eval/protocol.py. Task 1.1 (Tvisha, define the common
schema formally + the stratified split) has not landed on main yet, so this
module defines the fields locally as a plain dict/dataclass, matching the
schema string already written into ROADMAP.md:

    id, text, label, source, attack_type, domain, length

When 1.1 lands, these loaders should import the real schema from
eval/protocol.py instead of this local stand-in, and this file's `Row`
should be deleted in favor of that one. Keeping it here now (rather than
blocking on 1.1) is what makes 2.2 unblockable before 1.1 finishes.

Field meanings (inferred from ROADMAP.md's own field names + how the other
loaders -- data/generate_injection_authority.py, data/generate_perplexity_docs.py
-- are organized):
  id          : stable string ID, unique within the source file at minimum,
                e.g. "deepset_0001"
  text        : the raw document/prompt text
  label       : 0 = clean/benign, 1 = poisoned/injected (binary; matches the
                0/1 convention already used in eval/stats.py's compute_metrics)
  source      : which upstream dataset this row came from, e.g. "deepset",
                "opi", "hackaprompt"
  attack_type : free-text category, e.g. "prompt_injection", "spam_combine",
                "jailbreak" -- "clean" for label == 0 rows
  domain      : free-text content domain if known (e.g. "general",
                "sentiment_analysis", "spam_detection"), else "unknown"
  length      : len(text) in characters, computed automatically
"""

from dataclasses import dataclass, asdict


@dataclass
class Row:
    id: str
    text: str
    label: int
    source: str
    attack_type: str
    domain: str
    length: int

    @staticmethod
    def make(id, text, label, source, attack_type, domain="unknown"):
        return Row(
            id=id,
            text=text,
            label=int(label),
            source=source,
            attack_type=attack_type,
            domain=domain,
            length=len(text),
        )

    def to_dict(self):
        return asdict(self)


def print_source_counts(rows, source_name):
    """Satisfies the 2.2 'done when' criterion: counts printed per source."""
    n_total = len(rows)
    n_clean = sum(1 for r in rows if r.label == 0)
    n_poisoned = n_total - n_clean
    print(f"[{source_name}] {n_total} rows total "
          f"({n_clean} clean / {n_poisoned} poisoned)")
    by_attack = {}
    for r in rows:
        by_attack[r.attack_type] = by_attack.get(r.attack_type, 0) + 1
    for attack_type, count in sorted(by_attack.items()):
        print(f"  - {attack_type}: {count}")
