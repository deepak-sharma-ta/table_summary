# Table Summarizer - ToTTo data
"""The script for training a table summarizer
"""

import os
import subprocess
import yaml
import pandas as pd
import itertools
import glob
import json
import os
import subprocess
import yaml
from transformers import pipeline

from table_summary.log_config import LOGGING_DEFAULT_CONFIG, configure_logger
from table_summary.utils import generate_columns, generate_jsonl, create_table

with open("config.yaml", "r") as f:
    CONFIG = yaml.load(f, Loader=yaml.Loader)

DATA_PATH = CONFIG.get("DATA_PATH")
raw_data = DATA_PATH.get("raw_data")
processed_data = DATA_PATH.get("processed_data")
folder_name = DATA_PATH.get("folder_name")
TRAIN = DATA_PATH.get("train_data")
VAL = DATA_PATH.get("val_data")
TEST = DATA_PATH.get("test_data")

OUTPUT_DIR = CONFIG.get("MODEL").get("OUTPUT_DIR")

MODEL_PREDICTION = CONFIG.get("MODEL_PREDICTION").get("PATH")


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

    def pre_process(self, data=raw_data, test=False):
        random_num = 111234
        folder_list = glob.glob(os.path.join(data, f"{folder_name}/*"))
        json_file = []
        for folder in folder_list:
            file_list = glob.glob(folder + "/*.csv")  # ext placeholder
            if file_list:
                for file in file_list:
                    temp_json = {}
                    df = pd.read_csv(file)
                    if test:
                        if df.shape[0] > 100:
                            continue
                    df.columns = [column.replace("_", " ") for column in df.columns]

                    parent_list = create_table(df)  # list of dict
                    highlighted_cell = list(
                        itertools.product(range(1, df.shape[0] + 1), range(df.shape[1]))
                    )
                    highlighted_cells = [list(i) for i in highlighted_cell]

                    temp_json["table"] = parent_list
                    temp_json["table_webpage_url"] = ""
                    temp_json["table_page_title"] = ""
                    temp_json["table_section_title"] = ""
                    temp_json["table_section_text"] = ""
                    temp_json["highlighted_cells"] = highlighted_cells
                    temp_json["example_id"] = int(random_num)
                    temp_json["sentence_annotations"] = [
                        {
                            "original_sentence": "",
                            "sentence_after_deletion": "",
                            "sentence_after_ambiguity": "",
                            "final_sentence": "test",
                        }
                    ]

                    random_num += 1
                    json_file.append(temp_json)

        with open(os.path.join(processed_data, "unseen_data.jsonl"), "w") as outfile:
            for entry in json_file:
                json.dump(entry, outfile)
                outfile.write("\n")

        subprocess.run(
            ["git", "clone", "https://github.com/luka-group/Lattice.git"],
            check=True,
            capture_output=True,
        )
        self.logger.info("Cloned Lattice...")

        subprocess.run(
            ["pip", "install", "-r", "Lattice/requirements.txt"],
            check=True,
            capture_output=True,
        )
        self.logger.info("Installed the requirements for Lattice...")

        process_params = [
            "python",
            "Lattice/preprocess/preprocess_data.py",
            "--input_path",
            os.path.join(processed_data, "unseen_data.jsonl"),
            "--output_path",
            os.path.join(processed_data, "data_linearized.jsonl"),
        ]

        subprocess.run(
            process_params,
            check=True,
            capture_output=True,
        )
        self.logger.info("Executed Lattice...")

        lattice_output = pd.read_json(
            os.path.join(processed_data, "data_linearized.jsonl"), lines=True
        )  # jsonl is the output from lattice step 1

        model_input = pd.DataFrame()
        if "sentence_annotations" in lattice_output.columns:
            model_input["text"] = lattice_output["subtable_metadata_str"]
            model_input["summary"] = (
                lattice_output["sentence_annotations"]
                .apply(lambda x: x[0]["final_sentence"])
                .values
            )
        else:
            model_input["context"] = lattice_output["subtable_metadata_str"]
            model_input["summary"] = " "

        model_input.to_json(
            os.path.join(processed_data, "input_data.json"),
            orient="records",
            lines=True,
        )  # you can pass this input_data.json to the model
        self.logger.info("Pre-processing done...")

    def get_summarizer(self, summary_type: str = CONFIG.get("MODEL").get("INPUT_TYPE")):
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
            self.summary_type = summary_type.lower()

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
        model_name_or_path = CONFIG.get("MODEL").get(
            "PRETRAINED_MODEL"
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

        self.logger.info("Trainer started...")
        # Start the training
        trainer = subprocess.run(
            model_params,
            # shell=True
            check=True,
            capture_output=True,
        )
        self.logger.info(f"Trainer output: \n\n{trainer.stdout}")
        self.logger.info("Training done...")

        if self.summary_type == "table":
            self.predict(
                model_name_or_path=output_dir,
                output_dir=MODEL_PREDICTION,
                test_data=self.train_data,
                is_train=True,
            )
        else:
            self.predict(
                model_name_or_path=output_dir,
                output_dir=MODEL_PREDICTION,
                context=self.train_data,
                is_train=True,
            )

        self.logger.info("Summary generated for training data")

    def predict(
        self,
        model_name_or_path: str = None,
        output_dir: str = None,
        context: str = None,
        test_data: str = None,
        val_data: str = None,
        is_train=False,
    ):
        # getting the model path
        if not model_name_or_path:
            model_name_or_path = CONFIG.get("MODEL").get("FINETUNED_MODEL")

        if not output_dir:
            output_dir = MODEL_PREDICTION

        if self.summary_type == "text":
            if not context:
                raise TypeError("Please enter a valid context")
            summarizer = pipeline("summarization", model=model_name_or_path)

            if isinstance(context, str):
                generated_sum = summarizer(str(context))[0]["summary_text"]
                df = pd.DataFrame([generated_sum], columns=["predicted_summary"])
                df["context"] = context

            elif (
                isinstance(context, list)
                | isinstance(context, pd.Series)
                | isinstance(context, pd.DataFrame)
            ):
                generated_sum = []
                contexts = []
                for c in context:
                    generated_sum.append(summarizer(str(c))[0]["summary_text"])
                    contexts.append(c)

                df = pd.DataFrame(generated_sum, columns=["predicted_summary"])
                df["context"] = contexts

            df = generate_columns(
                df,
                model_name=model_name_or_path,
                input_type=self.summary_type,
                data_type="train" if is_train else "test",
            )

            # re-arranging the columns
            column_order = [
                "context",
                "predicted_summary",
                "model_name",
                "input_type",
                "data_type",
            ]

            df = df[column_order]

            out = model_name_or_path.replace("/", "_")
            if not out.endswith("_"):
                out += "_"

            df.to_json(
                os.path.join(output_dir, f"{out}prediction.jsonl"),
                orient="records",
            )
            self.logger.info(f"Summary generated output: \n\n{generated_sum}")
            self.logger.info(f"\nInference done for model: \n\n{model_name_or_path}")

        else:
            if not test_data:
                test_data = self.test_data
            if not val_data:
                val_data = self.val_data

            if not test_data and not val_data:
                raise TypeError(
                    "Please enter a valid path for test and validation data"
                )

            # get the model path
            model_params = [
                "python",
                "transformers/examples/pytorch/summarization/run_summarization.py",
                "--model_name_or_path",
                model_name_or_path,
                "--text_column",
                "text",
                "--overwrite_output_dir",  # to overwrite the existing files
                "--output_dir",
                output_dir,
                "--per_device_train_batch_size=4",
                "--per_device_eval_batch_size=4",
            ]
            # command line argument for predict
            predict_cli = [
                "--do_predict",
                "--test_file",
                test_data,
                "--predict_with_generate",
            ]
            # command line argument for validation
            validation_cli = [
                "--do_eval",
                "--validation_file",
                val_data,
            ]

            if test_data:  # if test data path is given
                model_params += predict_cli

            if val_data:  # if test val path is given
                model_params += validation_cli

            if model_name_or_path == "t5-small":  # if model name is t5 small
                model_params += self.t5_params

            # Start the training
            evaluator = subprocess.run(
                model_params,
                # shell=True
                check=True,
                capture_output=True,
            )
            generate_jsonl(
                test_data,
                output_dir,
                model_name_or_path,
                self.summary_type,
                "train" if is_train else "test",
            )

        self.logger.info(f"Evaluation output: \n\n{evaluator.stdout}")
        self.logger.info("Evaluation done...")
