"""Download and load NCAA tournament data from Kaggle."""

from pathlib import Path
import zipfile
import pandas as pd
import requests


KAGGLE_DATASETS = {
    "teams": "MTeams.csv",
    "seeds": "MNCAATourneySeeds.csv",
    "tourney_compact": "MNCAATourneyCompactResults.csv",
    "tourney_detailed": "MNCAATourneyDetailedResults.csv",
    "regular_detailed": "MRegularSeasonDetailedResults.csv",
    "regular_compact": "MRegularSeasonCompactResults.csv",
    "massey": "MMasseyOrdinals.csv",
    "conferences": "MTeamConferences.csv",
}


def download_kaggle_data(competition: str, output_dir: str | Path) -> None:
    """Download Kaggle competition data using the kaggle API.

    Requires ~/.kaggle/kaggle.json credentials file.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        api.competition_download_files(competition, path=str(output_dir))

        # Unzip any downloaded zip files
        for zf in output_dir.glob("*.zip"):
            with zipfile.ZipFile(zf, "r") as z:
                z.extractall(output_dir)
            zf.unlink()

        print(f"Downloaded and extracted data to {output_dir}")
    except ImportError:
        print("kaggle package not installed. Install with: pip install kaggle")
        print("Then place your API key at ~/.kaggle/kaggle.json")
        raise
    except Exception as e:
        print(f"Error downloading data: {e}")
        print("You can manually download from:")
        print(f"  https://www.kaggle.com/competitions/{competition}/data")
        raise


def load_dataset(name: str, data_dir: str | Path = "data/raw") -> pd.DataFrame:
    """Load a named dataset from the raw data directory."""
    if name not in KAGGLE_DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(KAGGLE_DATASETS.keys())}")

    filepath = Path(data_dir) / KAGGLE_DATASETS[name]
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found at {filepath}. Run `marchmadness download` first."
        )

    return pd.read_csv(filepath)


def load_all_datasets(data_dir: str | Path = "data/raw") -> dict[str, pd.DataFrame]:
    """Load all available datasets."""
    datasets = {}
    data_dir = Path(data_dir)
    for name, filename in KAGGLE_DATASETS.items():
        filepath = data_dir / filename
        if filepath.exists():
            datasets[name] = pd.read_csv(filepath)
    return datasets
