# NOTE: The below steps to execution are subject to change

# ToTTo Dataset

ToTTo is an open-domain English table-to-text dataset with over 120,000 training examples that proposes a controlled generation task: given a Wikipedia table and a set of highlighted table cells, produce a one-sentence description. https://github.com/google-research-datasets/totto

During the dataset creation process, tables from English Wikipedia are matched with (noisy) descriptions. Each table cell mentioned in the description is highlighted and the descriptions are iteratively cleaned and corrected to faithfully reflect the content of the highlighted cells.

## To execute the script
Create a virtual environment:  
    -> virtualenv .venv <br>

Activate the virtual environment:
    -> source .venv/bin/activate

Install the requirements:
    -> pip install -r deploy/dummy_req.txt # Place holders for now
    -> pip install transformers/.

Invoke the training:
    -> python src/trainer.py