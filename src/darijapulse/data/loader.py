"""Data loader and dataset manager for Moroccan Darija sentiment corpus."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Real public dataset hosted on Hugging Face
HF_MSAC_PARQUET_URL = (
    "https://huggingface.co/datasets/AbderrahmanSkiredj1/MSAC_darija_sentiment_analysis"
    "/resolve/main/data/train-00000-of-00001-f54b1f4d48309303.parquet"
)

# Diverse Arabizi validation test set for transliteration benchmarking
ARABIZI_BENCHMARK_EXAMPLES: list[dict[str, str | int]] = [
    {"text": "tbarkellah 3lik akhay service top w zwin bzaf", "label": 1},
    {"text": "merci bzaf 3la lmo3amala ltiyba bghit nchkor l equipe kamla", "label": 1},
    {"text": "had lproduit wa3er bzaf w kaykhdem mzyan", "label": 1},
    {"text": "lah yjazikom bikhir kolchi wsel f lwa9t w n9i", "label": 1},
    {"text": "saraha 3jebni bzaf ghadi n3awed nchri men 3ndkom", "label": 1},
    {"text": "application zwina sahla f lesti3mal bravo", "label": 1},
    {"text": "khedma n9iya bzaf w nas me2adbin tbarklah", "label": 1},
    {"text": "3jebni hadchi bzaf khdemto b nniya", "label": 1},
    {"text": "produit ghali w ma fih ta fayda khayeb bzaf", "label": 0},
    {"text": "makayn walo ma khedamch w lmo3amala na9sa bzaf", "label": 0},
    {"text": "service 3iyan w retard kbir makrehtch nred floussi", "label": 0},
    {"text": "had sel3a mghezla w khayba matshriwhach nas", "label": 0},
    {"text": "chffara ma kayjawbouch f telephone khedma 3yana", "label": 0},
    {"text": "tsenayt simana w f lekher wselni khaser ma 3jebnich", "label": 0},
    {"text": "khesart floussi 3la lkhawi maty9ohomch", "label": 0},
    {"text": "kolchi fih machakil application katbloqui dima", "label": 0},
]


class DarijaDataLoader:
    """Manages ingestion, caching, and preprocessing of Moroccan Darija sentiment datasets."""

    def __init__(self, data_dir: Path | str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.sample_dir = self.data_dir / "sample"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir.mkdir(parents=True, exist_ok=True)

    def load_msac_dataset(self, use_cache: bool = True) -> pd.DataFrame:
        """Download or read cached MSAC (Moroccan Sentiment Analysis Corpus) dataset."""
        cache_path = self.raw_dir / "msac_darija_sentiment.parquet"
        sample_path = self.sample_dir / "darija_sample_200.csv"

        if use_cache and cache_path.exists():
            logger.info("Loading cached MSAC Darija dataset from %s", cache_path)
            return pd.read_parquet(cache_path)

        # Attempt to load from remote Hugging Face parquet URL
        try:
            logger.info("Downloading MSAC Darija dataset from Hugging Face...")
            df = pd.read_parquet(HF_MSAC_PARQUET_URL)

            # Clean and normalize labels
            df.columns = [c.lower().strip() for c in df.columns]
            df["text"] = df["text"].astype(str).str.strip()
            df = df[df["text"].str.len() > 3].copy()

            # Map labels: pos -> 1, neg -> 0
            label_map = {"pos": 1, "neg": 0, ",ne": 0}
            df["label"] = df["label"].astype(str).str.lower().map(label_map).fillna(0).astype(int)

            df.reset_index(drop=True, inplace=True)
            df.to_parquet(cache_path, index=False)
            logger.info("Saved %d cleaned rows to cache %s", len(df), cache_path)

            # Create lightweight offline sample
            sample_df = df.sample(n=min(200, len(df)), random_state=42).copy()
            sample_df.to_csv(sample_path, index=False)
            logger.info("Saved sample to %s", sample_path)

            return df

        except Exception as exc:
            logger.warning("Failed to download from Hugging Face (%s). Falling back to sample or synthetic.", exc)
            if sample_path.exists():
                return pd.read_csv(sample_path)
            return self._generate_fallback_dataset()

    def load_arabizi_eval_set(self) -> pd.DataFrame:
        """Load curated Arabizi test set for bilingual model verification."""
        return pd.DataFrame(ARABIZI_BENCHMARK_EXAMPLES)

    def _generate_fallback_dataset(self) -> pd.DataFrame:
        """Generate deterministic bilingual Moroccan Darija dataset for offline testing."""
        positives = [
            "تبارك الله عليك خويا خدمة نقية بزاف وسيرفيس طوب",
            "منتوج رائع وثمن مناسب ووصل في الوقت المحدد شكرا",
            "هاد التطبيق زوين بزاف وساهل في الاستعمال برافو",
            "صراحة عجبني بزاف تعامل راقي والسلعة ممتازة",
            "الله يحفظكم معاملة في المستوى ومصداقية عالية",
            "tbarkellah 3lik khedma n9iya bzaf w service top",
            "had lproduit wa3er bzaf w taman munasib",
            "merci bzaf 3la lmo3amala ltiyba bghit nchkor equipe",
            "saraha 3jebni bzaf ghadi n3awed nchri men 3ndkom",
            "application zwina bzaf sahla bravo 3likom",
        ] * 20

        negatives = [
            "سلعة عيانة بزاف ومكايناش مصداقية ما ننصح تا حد يشري",
            "خدمة ناقصة وتعطلو عليا سيمانة كاملة ما كاين والو",
            "خسرت فلوسي على الخاوي هادشي كلو كذوب ونصب",
            "تطبيق ما خدامش ديما كيبلوكي وفيه بزاف ديال المشاكل",
            "ما عجبنيش الحال تعامل خايب ومكيردوش في التيليفون",
            "produit ghali w khayeb bzaf ma fih fayda",
            "service 3iyan w retard kbir makayn walo",
            "chffara ma kayjawbouch f telephone khedma na9sa",
            "tsenayt bzaf w f lekher wselni khaser ma 3jebnich",
            "khesart floussi 3la lkhawi maty9ohomch",
        ] * 20

        records = [{"text": t, "label": 1} for t in positives] + [{"text": t, "label": 0} for t in negatives]
        df = pd.DataFrame(records).sample(frac=1.0, random_state=42).reset_index(drop=True)
        return df
