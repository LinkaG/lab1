from hydra.utils import instantiate
import click
import pandas as pd
from joblib import dump
from ruamel.yaml import YAML
import mlflow
import os
from dotenv import load_dotenv


yaml = YAML(typ="safe")
load_dotenv()


@click.command()
@click.argument("input_path_data", type=click.Path())
@click.argument("input_path_label", type=click.Path())
@click.argument("output_path", type=click.Path())
def train(input_path_data: str, input_path_label: str, output_path: str):
    """Function to train model.
    :param input_path_data:Path to train dataset
    :param input_path_label: Path to train labels
    :param output_path: Path to model file
    :return:
    """
    mlflow.sklearn.autolog()
    mlflow.set_experiment("logreg classifier")
    mlflow.start_run(run_name="train_logreg")

    params = yaml.load(open("params.yaml", encoding="utf-8"))['model']['params']

    x_train = pd.read_csv(input_path_data)
    y_train = pd.read_csv(input_path_label)

    model = instantiate(params)
    model.fit(x_train, y_train.values.ravel())
    dump(model, output_path)


if __name__ == "__main__":
    train()
