"""Fetch and cache Moroccan Darija sentiment dataset."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.darijapulse.data.loader import DarijaDataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fetch_data")


def main() -> None:
    data_dir = PROJECT_ROOT / "data"
    loader = DarijaDataLoader(data_dir=data_dir)

    logger.info("Fetching and caching Moroccan Darija MSAC dataset...")
    df = loader.load_msac_dataset(use_cache=False)

    logger.info("Successfully loaded %d records.", len(df))
    logger.info("Class distribution:\n%s", df["label"].value_counts())

    arabizi_df = loader.load_arabizi_eval_set()
    logger.info("Curated Arabizi benchmark evaluation set contains %d examples.", len(arabizi_df))
    logger.info("Data caching complete.")


if __name__ == "__main__":
    main()
