# Use the Windows certificate store: the BEIR server omits an intermediate
# certificate that Python's bundled certificates cannot fill in on their own.
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass
"""
Tiny download-and-cache helper shared by the loaders.

Files land in data/external/ (git-ignored: we do NOT redistribute other
people's datasets, we only ship the code that fetches them). A file that is
already there is never downloaded again.
"""

import os
import urllib.request

EXTERNAL_DIR = os.environ.get("PDS_EXTERNAL_DIR", "data/external")


def fetch(url, relpath, base=None):
    dest = os.path.join(base or EXTERNAL_DIR, relpath)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    print(f"  downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "pds-research/0.1"})
    done, next_report = 0, 50 << 20
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as out:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if done >= next_report:
                print(f"    ... {done >> 20} MB")
                next_report += 50 << 20
    os.replace(tmp, dest)
    return dest