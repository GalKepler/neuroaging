from pathlib import Path

from loguru import logger
from tqdm import tqdm

from neuroaging.config import PROCESSED_DATA_DIR


def generate_features(input_path: Path, output_path: Path) -> None:
    logger.info("Generating features from dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Features generation complete.")


def main(
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
) -> None:
    generate_features(input_path=input_path, output_path=output_path)


if __name__ == "__main__":
    import typer

    typer.run(main)
