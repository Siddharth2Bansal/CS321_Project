import click  # Use the click library to provide a CLI interface
import importlib
import os

from pygarl.plugins.plist import list_serial_ports
from pygarl.plugins.record import record_new_samples
from pygarl.plugins.train import train_svm_classifier
from pygarl.plugins.sprint import sprint as sprint_func


def get_default_record_directory():
    """
    Used to generate the default user directory for saving new samples
    """
    # Get the user HOME directory
    home = os.path.expanduser("~")

    # Generate the complete path as: $HOME/dataset
    complete_path = os.path.join(home, "dataset")

    return complete_path


@click.group()
def cli():
    pass


@cli.command()
@click.option('--port', '-p', default="COM6", help="Serial Port NAME, for example COM3.")
@click.option('--dir', '-d', default=get_default_record_directory(),
              help="Target directory where samples will be saved.")
@click.option('--gesture', '-g', default="SAMPLE",
              help="Gesture ID of the recorded samples.")
@click.option('--axis', '-a', default=6, help="Number of AXIS in the signal, default 6.")
@click.option('--mode', '-m', default="discrete", help="Recording mode. You can choose discrete.")
@click.option('--threshold', '-t', default=40, help="If stream or piezo mode is specified, use this parameter to"
                                                            "regulate the threshold for the input.")
def record(port, dir, gesture, axis, mode, threshold):
    """
    Record new samples and saves them to file
    """
    if mode == "discrete":
        record_new_samples(port=port, gesture_id=gesture, target_dir=dir, expected_axis=axis)


@cli.command()
def plist():
    """
    Prints all the available serial ports
    """
    list_serial_ports()


@cli.command()
@click.option('--port', '-p', default="COM6", help="Serial Port NAME, for example COM3.")
@click.option('--baudrate', '-b', default=9600, help="Serial Port Baudrate, default 9600.")
def sprint(port, baudrate):
    """
    Print the input received from the specified serial port
    """
    sprint_func(port=port, baudrate=baudrate)


@cli.command()
@click.option('--dir', '-d', default=get_default_record_directory(),
              help="Dataset directory where samples are saved.")
@click.option('--classifier', '-c', default="svm",
              help="Classifier used to create a model. Default is SVM.")
@click.option('--trainer', '-t', default=None,
              help="Load a custom trainer. --classifier custom must be specified.")
@click.argument('output_file')
def train(dir, classifier, output_file, trainer):
    """
    Train a model from a dataset
    """
    # Load the appropriate method based on the specified classifier
    if classifier == "svm":
        train_svm_classifier(dir, output_file)
    else:
        raise ValueError("{classifier} is not a valid classifier".format(classifier=classifier))



if __name__ == '__main__':
    cli()
