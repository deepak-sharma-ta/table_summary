import pandas as pd
import itertools
import glob
import json
import os
import subprocess
import yaml

from table_summary.log_config import configure_logger, LOGGING_DEFAULT_CONFIG

LOG_FILE = None
CONSOLE_LOG = True

with open("config.yaml", "r") as f:
    CONFIG = yaml.load(f, Loader=yaml.Loader)

DATA_PATH = CONFIG.get("DATA_PATH")
raw_data = DATA_PATH.get("raw_data")
processed_data = DATA_PATH.get("processed_data")


configure_logger(
    log_file=os.path.join(LOG_FILE, "custom_config.log")
    if LOG_FILE
    else "logs/custom_config.log",
    console=CONSOLE_LOG,
    log_level=LOGGING_DEFAULT_CONFIG["root"]["level"],
)


def create_table(df):
    """
    convert unput df into jsonl file"""
    columns = list(df.columns)
    child_list = []
    for column in columns:
        child_list.append(
            {"value": column, "is_header": True, "column_span": 1, "row_span": 1}
        )

    # source & desn
    def iterate(row):
        output_list = []
        for column in columns:
            output_list.append(
                {
                    "value": str(row[column]),
                    "is_header": False,
                    "column_span": 1,
                    "row_span": 1,
                }
            )
        return output_list

    new_list = df.apply(iterate, axis=1).tolist()
    new_list.insert(0, child_list)

    return new_list


random_num = 111234

folder_list = glob.glob(os.path.join(raw_data, "extracted_folders/*"))
json_file = []
for folder in folder_list:
    file_list = glob.glob(folder + "/*.csv")  # ext placeholder
    if file_list:
        for file in file_list:
            # input_df = pd.DataFrame()
            temp_json = {}
            df = pd.read_csv(file)
            # if df.shape[0] > 100:
            #     continue
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


subprocess.run("git clone https://github.com/luka-group/Lattice.git")
subprocess.run("pip install -r Lattice/requirements.txt")

model_params = [
    "python",
    "Lattice/preprocess/preprocess_data.py ",
    "--input_path",
    os.path.join(processed_data, "unseen_data.jsonl"),
    "--output_path",
    os.path.join(processed_data, "data_linearized.jsonl"),
]


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
    os.path.join(processed_data, "input_data.json"), orient="records", lines=True
)  # you can pass this input_data.json to the model
