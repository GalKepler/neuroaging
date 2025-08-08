from pathlib import Path

from loguru import logger
from tqdm import tqdm

from neuroaging.config import PROCESSED_DATA_DIR, RAW_DATA_DIR


def process_dataset(input_path: Path, output_path: Path) -> None:
    logger.info("Processing dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Processing dataset complete.")


def main(
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
) -> None:
    process_dataset(input_path=input_path, output_path=output_path)


if __name__ == "__main__":
    import typer

    typer.run(main)
