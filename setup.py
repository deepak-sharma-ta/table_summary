from setuptools import setup

setup(
    name="table_summary",
    version="0.0.1",
    # package_dir={"": "table_summarizer"},
    install_requires=[
        "datasets",
        "evaluate",
        "rouge_score",
        "argparse",
        "pandas",
    ],
)
