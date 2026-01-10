from pathlib import Path

from loguru import logger
from tqdm import tqdm

from neuroaging.config import MODELS_DIR, PROCESSED_DATA_DIR


def train_model(features_path: Path, labels_path: Path, model_path: Path) -> None:
    logger.info("Training some model...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Modeling training complete.")


def main(
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
) -> None:
    train_model(features_path=features_path, labels_path=labels_path, model_path=model_path)


if __name__ == "__main__":
    import typer

    typer.run(main)
