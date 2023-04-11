# Table Summarizer - ToTTo data
"""The script for training a table summarizer
"""

import os
import sys
import yaml
import subprocess
import pandas as pd
import argparse

sys.path.append("/home/deepak/table_summary")
from table_summarizer.log_config import LOGGING_DEFAULT_CONFIG, configure_logger


# with open("config.yaml", "r") as f:
#     CONFIG = yaml.load(f, Loader=yaml.Loader)

# DATA_PATH = CONFIG.get("DATA_PATH")
# TRAIN_DATA = DATA_PATH.get("train_data")
# TEST_DATA = DATA_PATH.get("test_data")
# VAL_DATA = DATA_PATH.get("val_data")
# TARGET_DATA = DATA_PATH.get("totto_target_summary")
# ARTIFACTS = DATA_PATH.get("ARTIFACTS")
# GENERATED_PREDICTION = DATA_PATH.get("GENERATED_PREDICTION")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--no-console-log",
    action="store_true",
    required=False,
    help="add the path of log level",
)
parser.add_argument(
    "-lp",
    "--log-path",
    type=str,
    required=False,
    help="add the path of log level",
)

args = parser.parse_args()
LOG_FILE = None
if args.log_path:
    LOG_FILE = args.log_path

CONSOLE_LOG = True
if args.no_console_log:
    CONSOLE_LOG = False


class Summarizer:
    def __init__(self):
        self.log_file_path = LOG_FILE
        self.logger = configure_logger(
            log_file=os.path.join(LOG_FILE, "custom_config.log")
            if LOG_FILE
            else "logs/custom_config.log",
            console=CONSOLE_LOG,
            log_level=LOGGING_DEFAULT_CONFIG["root"]["level"],
        )
        self.summary_type = None
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def get_summarizer(self, summary_type: str = None):
        """
        Initialize the sumamrizer with type of summary expected

        Parameters
        ----------
        summary_type : str
            Type of summary input
        """
        if summary_type != "text" or summary_type != "table":
            raise TypeError(
                "Please enter a value for summary type to be either text or table"
            )
        else:
            self.summary_type = summary_type

    def load_data(self, train_path: str, test_path: str = None, val_path: str = None):
        """
        Initialize the sumamrizer with type of summary expected

        Parameters
        ----------
        train_path : str
            Path to the training data

        test_path: str
            Path to the test data

        val_path: str
            Path to the validation data
        """
        self.logger.warning("Logging - Start")
        self.logger.info("Loading the data...")

        self.train_data = pd.read_csv(train_path)
        self.logger.info(f"Train sample size: {self.train_data.shape[0]}")

        if val_path:
            self.val_data = pd.read_csv(val_path)
            self.logger.info(f"Validation sample size: {self.val_data.shape[0]}")
        if test_path:
            self.test_data = pd.read_csv(test_path)
            self.logger.info(f"Test sample size: {self.test_data.shape[0]}")

    def train(self, output_dir):
        """
        Initialize the sumamrizer with type of summary expected

        Parameters
        ----------
        output_dir : str
            Path to export the trained model
        """
        # Start the training
        subprocess.run(
            [
                "python",
                "transformers/examples/pytorch/summarization/run_summarization.py",
                "--model_name_or_path t5-small",
                "--do_train",
                "--do_eval",
                f"--train_file {self.train_data}",
                f"--validation_file {self.val_data}",
                '--text_column "text"',
                '--summary_column "summary"',
                '--source_prefix "summarize: "',
                "--overwrite_output_dir",
                f"--output_dir {output_dir}",
                "--per_device_train_batch_size=4",
                "--per_device_eval_batch_size=4",
                "--predict_with_generate",
                "--max_train_samples {100}",  # optional
                "--max_predict_samples {50}",  # optional
            ],
            shell=True,
            check=True,
        )
        self.logger.info("Training done...")


# # after execution of the above code
# # output generated_predictions.txt will be stored in predictions/fine-tune/facebook-bart/

# # Cloning the language repo
# subprocess.run(
#     [
#         "git",
#         "clone",
#         "https://github.com/google-research/language.git",
#         "language_repo",
#     ],
#     shell=True,
#     check=True,
# )
# logger.info("Cloned language repo...")

# # Installing the requirements for language repo
# subprocess.run(
#     [
#         "pip3",
#         "install",
#         "-r",
#         "language_repo/language/totto/eval_requirements.txt",
#         "language_repo",
#     ],
#     shell=True,
#     check=True,
# )
# logger.info("Installed language repo requirements...")

# # Making predictions
# subprocess.run(
#     [
#         "bash",
#         "language_repo/language/totto/totto_eval.sh",
#         f"--prediction_path {GENERATED_PREDICTION.get('generated_txt')}",
#         f"--target_path {TARGET_DATA}",
#         f"--output_dir {CONFIG.get('OUTPUT_DIR')}",
#     ],
#     shell=True,
#     check=True,
# )
