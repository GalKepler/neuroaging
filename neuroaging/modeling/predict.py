from pathlib import Path

from loguru import logger
from tqdm import tqdm

from neuroaging.config import MODELS_DIR, PROCESSED_DATA_DIR


def predict_model(features_path: Path, model_path: Path, predictions_path: Path) -> None:
    logger.info("Performing inference for model...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Inference complete.")


def main(
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
) -> None:
    predict_model(
        features_path=features_path, model_path=model_path, predictions_path=predictions_path
    )


if __name__ == "__main__":
    import typer

    typer.run(main)
