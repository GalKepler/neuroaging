import typer

from .dataset import main as dataset_main
from .features import main as features_main
from .modeling.predict import main as predict_main
from .modeling.train import main as train_main
from .plots import main as plots_main

app = typer.Typer(help="Command-line interface for neuroaging.")

app.command("dataset")(dataset_main)
app.command("features")(features_main)
app.command("plots")(plots_main)
app.command("train")(train_main)
app.command("predict")(predict_main)


if __name__ == "__main__":
    app()
