from sklearn.model_selection import train_test_split
import click
import pandas as pd


@click.command()
@click.argument("input_path", type=click.Path())
@click.argument("output_path_train_data", type=click.Path())
@click.argument("output_path_train_labels", type=click.Path())
@click.argument("output_path_test_data", type=click.Path())
@click.argument("output_path_test_labels", type=click.Path())
def split_data(input_path: str,
               output_path_train_data: str,
               output_path_train_labels: str,
               output_path_test_data: str,
               output_path_test_labels: str):
    """ Split init dataset to train and test datasets.
    :param input_path: Path to init DataFrame
    :param output_path_train_data: Path to train DataFrame
    :param output_path_train_labels: Path to train labels DataFrame
    :param output_path_test_data: Path to test DataFrame
    :param output_path_test_labels: Path to test labels DataFrame
    :return:
    """

    df = pd.read_csv(input_path)
    X_train, X_test, y_train, y_test = train_test_split(df.drop('target', axis=1), df.target, test_size=0.33,
                                                        random_state=42)

    X_train.to_csv(output_path_train_data, index=False)
    X_test.to_csv(output_path_test_data, index=False)
    y_train.to_csv(output_path_train_labels, index=False)
    y_test.to_csv(output_path_test_labels, index=False)


if __name__ == "__main__":
    split_data()