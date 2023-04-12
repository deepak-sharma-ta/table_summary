# Table Summarizer - ToTTo data
"""The script for training a table summarizer
"""

import os
import sys
import yaml
import subprocess
import pandas as pd
import argparse

# sys.path.append("/home/deepak/table_summary")
from table_summarizer.log_config import LOGGING_DEFAULT_CONFIG, configure_logger


with open("config.yaml", "r") as f:
    CONFIG = yaml.load(f, Loader=yaml.Loader)

DATA_PATH = CONFIG.get("DATA_PATH")
TRAIN = DATA_PATH.get("train_data")
VAL = DATA_PATH.get("val_data")
TEST = DATA_PATH.get("test_data")
OUTPUT_DIR = CONFIG.get("MODEL").get("OUTPUT_DIR")
GENERATED_PREDICTION = CONFIG.get("GENERATED_PREDICTION").get("generated_txt")

LOG_FILE = None
CONSOLE_LOG = True


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
        self.t5_params = [
            "--source_prefix",
            "summarize: ",
        ]

    def get_summarizer(self, summary_type: str = None):
        """
        Initialize the sumamrizer with type of summary expected

        Parameters
        ----------
        summary_type : str
            Type of summary input
        """
        if summary_type != "text" and summary_type != "table":
            raise TypeError(
                "Please enter a value for summary type to be either text or table"
            )
        else:
            self.summary_type = summary_type

    def load_data(
        self, train_path: str = TRAIN, test_path: str = TEST, val_path: str = VAL
    ):
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

        self.train_data = train_path
        self.logger.info(f"Train sample size: {pd.read_csv(self.train_data).shape[0]}")

        if val_path:
            self.val_data = val_path
            self.logger.info(
                f"Validation sample size: {pd.read_csv(self.val_data).shape[0]}"
            )
        if test_path:
            self.test_data = test_path
            self.logger.info(
                f"Test sample size: {pd.read_csv(self.test_data).shape[0]}"
            )

    def train(self, output_dir=OUTPUT_DIR):
        """
        Initialize the sumamrizer with type of summary expected

        Parameters
        ----------
        output_dir : str
            Path to export the trained model
        """
        model_name_or_path = CONFIG.get(
            self.summary_type.upper().get("MODEL_NAME")
        )  # get the model type

        model_params = [
            "python",
            "transformers/examples/pytorch/summarization/run_summarization.py",
            "--model_name_or_path",
            model_name_or_path,
            "--do_train",
            "--do_eval",
            "--train_file",
            self.train_data,
            "--validation_file",
            self.val_data,
            "--text_column",
            "text",
            "--summary_column",
            "summary",
            "--overwrite_output_dir",  # to overwrite the existing files
            "--output_dir",
            output_dir,
            "--per_device_train_batch_size=4",
            "--per_device_eval_batch_size=4",
            "--predict_with_generate",
            "--max_train_samples",
            "100",  # optional
            "--max_predict_samples",
            "50",  # optional
        ]

        if model_name_or_path == "t5-small":
            model_params += self.t5_params

        # Start the training
        trainer = subprocess.run(
            model_params,
            # shell=True
            check=True,
            capture_output=True,
        )
        self.logger.info(f"Trainer output: \n\n{trainer.stdout}")
        self.logger.info("Training done...")

    def predict(self, output_dir: str):
        model_name_or_path = CONFIG.get("MODEL").get("MODEL_PATH")  # get the model path
        model_params = [
            "python",
            "transformers/examples/pytorch/summarization/run_summarization.py",
            "--model_name_or_path",
            model_name_or_path,
            "--do_predict",
            "--do_eval",
            "--test_file",
            self.test_data,
            "--validation_file",
            self.val_data,
            "--text_column",
            "text",
            "--overwrite_output_dir",  # to overwrite the existing files
            "--output_dir",
            output_dir,
            "--per_device_train_batch_size=4",
            "--per_device_eval_batch_size=4",
            "--predict_with_generate",
        ]

        if model_name_or_path == "t5-small":
            model_params += self.t5_params

        # Start the training
        evaluator = subprocess.run(
            model_params,
            # shell=True
            check=True,
            capture_output=True,
        )
        self.logger.info(f"Evaluation output: \n\n{evaluator.stdout}")
        self.logger.info("Evaluation done...")
