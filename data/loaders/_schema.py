"""
Shared row schema for all data/loaders/*.py modules.

The real definition now lives in eval/protocol.py (roadmap task 1.1). This
file just re-exports it so the existing loaders (deepset.py, opi.py,
hackaprompt.py) keep working unchanged.
"""

from eval.protocol import Row, print_source_counts  # noqa: F401