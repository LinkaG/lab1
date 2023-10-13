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
tracking_uri = os.environ.get("TRACKING_URI")


@click.command()
@click.argument("input_path_data", type=click.Path())
@click.argument("input_path_label", type=click.Path())
@click.argument("output_path", type=click.Path())
def train(input_path_data: str, input_path_label: str, output_path: str):
    """ Function to train smv-classifier.
    :param input_path_data: Path to train DataFrame
    :param input_path_label: Path to train labels DataFrame
    :param output_path: Path to trained model
    :return:
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.sklearn.autolog()
    mlflow.set_experiment("svm classifier")
    mlflow.start_run(run_name ="train_svm")

    params = yaml.load(open("params.yaml", encoding="utf-8"))['model']['params']

    x_train = pd.read_csv(input_path_data)
    y_train = pd.read_csv(input_path_label)

    model = instantiate(params)
    model.fit(x_train, y_train.values.ravel())
    dump(model, output_path)


if __name__ == "__main__":
    train()
