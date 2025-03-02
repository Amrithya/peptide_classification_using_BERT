# PeptideBERT
Transformer Based Language Model for Peptide Prediction.


## Getting Started
* Clone this repository
* Install the required packages (`pip install -r requirements.txt`)

<br>

## How to Use with new dataset
* Update the `config.py` file with the desired parameters
* Run `python data/split_augment.py` to convert the data into the required format
* Optionally, to augment the data, use `data/split_augment.py` (uncomment the line that calls `augment_data`)

## With exisiting dataset
* Run `python train.py` to train the model and `python predict.py` to predict on the test file.

Note: For a detailed walkthrough of the codebase (including how to run inference using a trained model), please refer to `tutorial.ipynb`.
