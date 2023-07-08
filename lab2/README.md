# Lab 2 - Natural Language Processing (NLP)

This laboratory is focused on the use of Language Models (LMs) for Natural Language Processing (NLP) tasks. The main goal is to introduce Language Models by getting the hands on GPT model, and then explore the 🤗 Transformers library, using LMs for text generation, Classification and  Multiple Choice Question Answering (MCQA).

[![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=plastic&logo=jupyter&logoColor=white)](./Lab2-LLMs.ipynb)

The code for all three exercises is available in the [notebook](./Lab2-LLMs.ipynb) file.

## 1. Introduction to Language Models - GPT
This section uses a pretrained GPT model to perform character-level language modeling. The model is trained on the [divina commedia](https://it.wikipedia.org/wiki/La_divina_commedia) by Dante Alighieri.

## 2. Text Generation with 🤗 Transformers
In this section we introduce the 🤗 Transformers library and use it to perform text generation, showing the main techniques to perform this task.

## 3. Classification and MCQA with 🤗 Transformers and⚡Lightning 

[![WandB report](https://img.shields.io/badge/Weights_&_Biases-FFCC33?style=plastic&logo=WeightsAndBiases&logoColor=black)](https://api.wandb.ai/links/dla-darcio/ibxzhjmp)

In this section we employed pretrained models from the 🤗 Transformers library to perform Classification and MCQA tasks. We used the [Tweet_eval](https://arxiv.org/abs/2010.12421v2) dataset for classification and the [Swag](https://arxiv.org/abs/1808.05326) dataset for MCQA.

For both tasks, we extracted the features using a pretrained DistilBERT model and then trained a simple MLP on top of the features. We also trained the DistilBERT model on top of the features for the MCQA task.

The results are also available on [WandB](https://api.wandb.ai/links/dla-darcio/ibxzhjmp).

### Usage
To extract features from a MCQA dataset, run

```bash
python feature_qa_feature_extraction.py --dataset <dataset_name> --model <model_name> --batch_size <batch_size>
```

where `<dataset_name>` can be `swag` or `race`, `<model_name>` can be any language model from Huggingface and `<batch_size>` is the batch size used for feature extraction.

To train a model on pre-extracted features, run

```bash
python main.py fit -c configs/<config_file>.yaml
```

where `<config_file>` is the name of the YAML configuration file. The configuration files are located in the `configs` directory.

To evaluate a model on pre-extracted features, run

```bash
python main.py test -c configs/<config_file>.yaml
```

where `<config_file>` is the name of the YAML configuration file. The configuration files are located in the `configs` directory.

## Requirements

The following packages are required to run the code in this repository.

- python 3.10
- pytorch 2.0.0
- torchvision 0.15.0
- wandb 0.15.0
- jupyterlab
- ipython
- matplotlib
- scikit-learn 1.2.2
- tqdm
- numpy
- transformers
- datasets 
- lightning 2.0.0

## Project Structure

```
lab-2/
│
├── configs/ - directory with example YAML configuration files
│ 
├── data/ - directory for storing input data
│   ├── swag/ - directory for swag dataset containing features extracted from train, validation and test splits
│   └── inferno.txt - training textfile for Exercise 1
│
├── runs/ - directory for local logging with Weights & Biases and checkpoint storage
│
├── models/ - directory of developed models
│   ├── gpt.py - gpt impementation for Exercise 1 taken from
│   ├── qa_mlp.py - simple MLP for MCQA on pre-extracted features
│   └── qa_transformer.py - Transformer model for MCQA
│
├── data_loader.py - anything about data loading goes here
│   ├── collator.py - custom collation for Multiple Choice Question Answering
│   ├── feature_datamodule.py - data module for loading extracted features
│   ├── swag_datamodule.py - data module for swag dataset
│   └── race_datamodule.py - data module for race dataset
│
├── feature_extraction.py - script for extracting features from MCQA datasets
└── main.py - main script for training/evaluation on pre-extracted features
```

