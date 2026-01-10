from pathlib import Path

from loguru import logger
from tqdm import tqdm

from neuroaging.config import FIGURES_DIR, PROCESSED_DATA_DIR


def generate_plot(input_path: Path, output_path: Path) -> None:
    logger.info("Generating plot from data...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Plot generation complete.")


def main(
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = FIGURES_DIR / "plot.png",
) -> None:
    generate_plot(input_path=input_path, output_path=output_path)


if __name__ == "__main__":
    import typer

    typer.run(main)
