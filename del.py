from table_summarizer.trainer import Summarizer

model = Summarizer()

model.get_summarizer(summary_type="table")
model.load_data(
    train_path="data/processed/sub_set_1.csv",
    val_path="data/processed/val_data_test.csv",
)
model.train(output_dir="test_output")
