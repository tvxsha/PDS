"""
Writes data/_provenance/corpus_sources.tsv for the 15 corpus-poisoning docs.

These docs contain NO instruction-override language. Each gives
plausible-sounding but insecure technical advice (a "security
anti-pattern"), so they are meant to slip past the pattern detector.

Run from the repo root:  python data/generate_corpus_provenance.py
"""

import os

DOCS = [
    ("corpus_001.txt", "disable_tls_verification_and_plaintext_password"),
    ("corpus_002.txt", "secrets_in_client_side_code"),
    ("corpus_003.txt", "disable_session_expiry_and_audit_log"),
    ("corpus_004.txt", "disable_certificate_verification"),
    ("corpus_005.txt", "social_engineering_password_request"),
    ("corpus_006.txt", "excessive_default_privileges"),
    ("corpus_007.txt", "credential_reuse_after_rotation"),
    ("corpus_008.txt", "hardcoded_root_password"),
    ("corpus_009.txt", "cors_wildcard_in_production"),
    ("corpus_010.txt", "auth_header_in_plaintext_logs"),
    ("corpus_011.txt", "skip_webhook_signature_verification"),
    ("corpus_012.txt", "long_lived_script_readable_cookie"),
    ("corpus_013.txt", "sensitive_data_shared_with_third_party"),
    ("corpus_014.txt", "ci_runner_as_root"),
    ("corpus_015.txt", "user_enumeration_via_error_messages"),
]

SOURCE = "synthetic"
METHOD = "hand_authored_security_antipattern"  # edit if these were written with LLM help


def main():
    out_path = "data/_provenance/corpus_sources.tsv"
    missing = [n for n, _ in DOCS if not os.path.exists(f"data/poisoned/{n}")]
    if missing:
        raise SystemExit(f"Missing files in data/poisoned/: {missing} -- run from the repo root.")

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write("filename\tsource\tmethod\tattack_subtype\n")
        for name, subtype in DOCS:
            f.write(f"{name}\t{SOURCE}\t{METHOD}\t{subtype}\n")
    print(f"Wrote {len(DOCS)} rows to {out_path}")


if __name__ == "__main__":
    main()