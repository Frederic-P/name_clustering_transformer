# Transformer based pairwise name similarity tool

## Installation:
Developped for python 3.12
1) create a Python3.12 venv: `python3.12 -m venv .venv`; or any other appropriate alias for python 3.12 you use on your system. 
2) activate: `source .venv/bin/activate`
3) install dependencies: `pip install -r requirements.txt`

## Training: 
Fine-tuning is done with the `training_notebooks/fine_tuning_AUTO.py` script it'll take any transformer model from huggingface that is compatible with *transformers.AutoModel* and *transformers.AutoTokenizer*. The goal of training is to optimize a general purpose transformer model to be able to make a pairwise based prediction on name similarity. 

GPU-acceleration is handled automatically for CUDA-platforms


## Inferring: 
see the `reporting/test_on_studium.ipynb` for inferring. 