import pandas as pd
import os


def generate_columns(
    data, model_name: str = None, input_type: str = None, data_type: str = None
):
    """
    add columns like model_name, input_type, data_type
    """
    if not model_name or not input_type or not data_type:
        raise TypeError(
            "Please provide proper values for model_name, input_type, data_type"
        )

    data["model_name"] = model_name
    data["input_type"] = input_type
    data["data_type"] = data_type

    return data


def generate_jsonl(data, prediction_path, model_name, summary_type, data_type):
    train_df = pd.read_csv(data)
    train_df.columns = (
        ["context", "actual_summary"] if summary_type == "table" else ["context"]
    )

    pred_sum = open(
        os.path.join(prediction_path, "generated_predictions.txt"), "r"
    ).readlines()
    pred_sum = [x.replace("\n", " ").strip() for x in pred_sum]
    pred_sum = pd.DataFrame(pred_sum, columns=["predicted_summary"])

    final_df = pd.concat([train_df, pred_sum], axis=1)
    final_df = generate_columns(
        final_df,
        model_name=model_name,
        input_type=summary_type,
        data_type=data_type,
    )
    final_df.to_json(
        os.path.join(prediction_path, "generated_predictions.jsonl"),
        orient="records",
    )


def create_table(data):
    """
    convert unput df into jsonl file"""
    columns = list(data.columns)
    child_list = []
    for column in columns:
        child_list.append(
            {
                "value": column,
                "is_header": True,
                "column_span": 1,
                "row_span": 1,
            }
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

    new_list = data.apply(iterate, axis=1).tolist()
    new_list.insert(0, child_list)

    return new_list
