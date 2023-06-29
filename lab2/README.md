# Lab 2 - Natural Language Processing (NLP)

This laboratory is focused on the use of Language Models (LMs) for Natural Language Processing (NLP) tasks. The main goal is to introduce Language Models by getting the hands on GPT model, and then explore the 🤗 Transformers library, using LMs for text generation, Classification and  Multiple Choice Question Answering (MCQA).

The code is available in the [notebook](./Lab2-LLMs.ipynb) file. The notebook is divided in three sections, each one with a different task.

## 1. Introduction to Language Models - GPT
This section uses a pretrained GPT model to perform character-level language modeling. The model is trained on the [divina commedia](https://it.wikipedia.org/wiki/La_divina_commedia) by Dante Alighieri.

## 2. Text Generation with 🤗 Transformers
In this section we introduce the 🤗 Transformers library and use it to perform text generation, showing the main techniques to perform this task.

## 3. MCQA with 🤗 Transformers and⚡Lightning 
In this section we employed pretrained models from the 🤗 Transformers library to perform MCQA tasks. We used the [Swag](https://arxiv.org/abs/1808.05326) dataset and [RACE](https://aclanthology.org/D17-1082.pdf) datasets, two Multiple Choice Question Answering datasets. Both dataset are composed by an  initial sentence describing the context, and four different options.

The procedure and the results are also available on WandB.



