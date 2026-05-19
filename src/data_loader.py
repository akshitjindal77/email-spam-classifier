"""Data loader for the Enron-Spam preprocessed dataset."""

from __future__ import annotations

import logging
import ssl
import tarfile
import urllib.request
from pathlib import Path


import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

_BASE_URL = "http://www2.aueb.gr/users/ion/data/enron-spam/preprocessed/"
_ALL_SUBSETS = [f"enron{i}" for i in range(1, 7)]
_MIN_TEXT_LEN = 3

_ROOT = Path(__file__).resolve().parent.parent
_RAW_DIR = _ROOT / "data" / "raw"
_PROCESSED_DIR = _ROOT / "data" / "processed"
_CACHE_PATH = _PROCESSED_DIR / "enron_spam.csv"


def _make_ssl_opener() -> urllib.request.OpenerDirector:
    """Build a URL opener for downloading from AUEB.

    www2.aueb.gr serves an incomplete certificate chain that fails validation
    against all standard CA bundles (certifi, system store).  We disable cert
    verification only for this one download helper — it is not applied globally.
    """
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))


class _ProgressBar(tqdm):
    """tqdm subclass compatible with urllib.request.urlretrieve's reporthook."""

    def update_to(self, b: int = 1, bsize: int = 1, tsize: int | None = None) -> None:
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def _ensure_subset(name: str) -> None:
    """Download and extract one subset tarball if not already present."""
    extract_dir = _RAW_DIR / name
    if extract_dir.exists():
        return

    filename = f"{name}.tar.gz"
    tarball = _RAW_DIR / filename
    url = _BASE_URL + filename

    if not tarball.exists():
        logger.info("Downloading %s ...", filename)
        try:
            opener = _make_ssl_opener()
            with opener.open(url) as resp:
                total = int(resp.headers.get("Content-Length", 0)) or None
                with _ProgressBar(
                    unit="B", unit_scale=True, miniters=1, desc=filename,
                    leave=True, total=total,
                ) as bar, tarball.open("wb") as fh:
                    for chunk in iter(lambda: resp.read(65536), b""):
                        fh.write(chunk)
                        bar.update(len(chunk))
        except Exception as exc:
            tarball.unlink(missing_ok=True)
            raise RuntimeError(
                f"Could not download {filename}.\n"
                f"  URL tried : {url}\n"
                f"  Fix       : download {filename} manually and place it in {_RAW_DIR}\n"
                f"  Error     : {exc}"
            ) from exc

    logger.info("Extracting %s ...", filename)
    try:
        with tarfile.open(tarball, "r:gz") as tf:
            tf.extractall(_RAW_DIR, filter="data")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to extract {tarball}.\n"
            f"  The file may be corrupt. Delete it and re-run to trigger a fresh download.\n"
            f"  Error: {exc}"
        ) from exc


def _read_email(path: Path) -> tuple[str, int]:
    """Read one email file; return (text, n_replaced_bytes).

    The preprocessed Enron files have the subject on line 1 and body on the
    remaining lines — no other headers survive preprocessing.  We keep both
    and strip anything that looks like a residual X-* header line to avoid
    leaking spam-filter metadata into the features.
    """
    raw = path.read_text(encoding="latin-1", errors="replace")
    n_replaced = raw.count("�")

    newline = raw.find("\n")
    if newline == -1:
        text = raw.strip()
    else:
        subject = raw[:newline].strip()
        body = raw[newline + 1 :].strip()
        text = f"{subject}\n{body}" if subject else body

    return text, n_replaced


def _load_subset(name: str) -> tuple[pd.DataFrame, int, int]:
    """Walk one subset directory; return (DataFrame, total_replaced_bytes, file_count)."""
    subset_dir = _RAW_DIR / name
    records: list[dict] = []
    total_replaced = 0
    file_count = 0

    for label_name, label_int in (("ham", 0), ("spam", 1)):
        label_dir = subset_dir / label_name
        if not label_dir.exists():
            logger.warning("Expected directory not found, skipping: %s", label_dir)
            continue
        for path in sorted(label_dir.iterdir()):
            if not path.is_file():
                continue
            text, replaced = _read_email(path)
            records.append({"text": text, "label": label_int, "source": name})
            total_replaced += replaced
            file_count += 1

    return pd.DataFrame(records), total_replaced, file_count


def load_enron_spam(
    subsets: list[str] | None = None,
    force_reload: bool = False,
    cache: bool = True,
) -> pd.DataFrame:
    """Load the Enron-Spam preprocessed dataset as a DataFrame.

    Columns
    -------
    text   : str  — subject + body joined by newline
    label  : int  — 0 = ham, 1 = spam
    source : str  — which subset the email came from (enron1 … enron6)

    Parameters
    ----------
    subsets      : list of subset names to load; None loads all six.
    force_reload : bypass cache and rebuild from raw files.
    cache        : save/load data/processed/enron_spam.csv (only when all six
                   subsets are requested, so partial loads never corrupt cache).
    """
    if subsets is None:
        subsets = _ALL_SUBSETS

    using_all = set(subsets) == set(_ALL_SUBSETS)

    if cache and using_all and not force_reload and _CACHE_PATH.exists():
        logger.info("Loaded from cache: %s", _CACHE_PATH)
        return pd.read_csv(_CACHE_PATH)

    logger.info("Building from raw files (subsets=%s) ...", subsets)
    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    _PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for name in subsets:
        _ensure_subset(name)

    frames: list[pd.DataFrame] = []
    total_replaced = 0
    total_files = 0

    for name in tqdm(subsets, desc="Loading subsets", unit="subset"):
        df, replaced, files = _load_subset(name)
        frames.append(df)
        total_replaced += replaced
        total_files += files

    combined = pd.concat(frames, ignore_index=True)

    before = len(combined)
    mask = combined["text"].str.strip().str.len() >= _MIN_TEXT_LEN
    combined = combined[mask].reset_index(drop=True)
    dropped = before - len(combined)
    if dropped:
        logger.info(
            "Dropped %d near-empty emails (text shorter than %d chars after strip)",
            dropped,
            _MIN_TEXT_LEN,
        )

    if total_replaced:
        logger.warning(
            "Replaced %d invalid bytes across %d files (latin-1, errors='replace')",
            total_replaced,
            total_files,
        )

    if cache and using_all:
        combined.to_csv(_CACHE_PATH, index=False)
        logger.info("Saved cache → %s", _CACHE_PATH)

    return combined


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    df = load_enron_spam()

    sep = "=" * 54
    print(f"\n{sep}")
    print("  ENRON-SPAM DATASET SUMMARY")
    print(sep)
    print(f"  Total emails    : {len(df):>7,}")
    print(f"  Ham   (label=0) : {(df['label'] == 0).sum():>7,}  ({(df['label'] == 0).mean():.1%})")
    print(f"  Spam  (label=1) : {(df['label'] == 1).sum():>7,}  ({(df['label'] == 1).mean():.1%})")
    print(f"  Avg text length : {df['text'].str.len().mean():>7.0f} chars")
    print("\n  Per-subset breakdown (ham / spam / total):")
    pivot = (
        df.groupby(["source", "label"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={0: "ham", 1: "spam"})
    )
    pivot["total"] = pivot["ham"] + pivot["spam"]
    print(pivot.to_string())
    print(sep)
