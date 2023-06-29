from typing import List, Union

import datasets
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data_loader.collator import DataCollatorForMultipleChoice


class SWAGDataModule(LightningDataModule):
    text_fields = "sent1"
    question_field = "sent2"
    ending_names = ['ending0', 'ending1', 'ending2', 'ending3']
    label_name = "label"

    num_labels = 4

    loader_columns = [
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "label",
    ]

    def __init__(
            self,
            model_name_or_path: str,
            task_name: str = "regular",
            max_seq_length: int = 128,
            train_batch_size: int = 32,
            eval_batch_size: int = 32,
            **kwargs,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def setup(self, stage: str):
        self.dataset = datasets.load_dataset("swag", self.task_name)

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns=self.dataset[split].column_names
            )
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]

    def prepare_data(self):
        datasets.load_dataset("swag", self.task_name)
        AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def convert_to_features(self, example_batch, indices=None):
        first_sentences = [[context] * 4 for context in example_batch[self.text_fields]]
        questions = example_batch[self.question_field]
        second_sentences = [
            [f"{header} {example_batch[end][i]}" for end in self.ending_names] for i, header in enumerate(questions)
        ]
        # Flatten the list
        first_sentences = sum(first_sentences, [])
        second_sentences = sum(second_sentences,[])

        text_pairs = list(zip(first_sentences, second_sentences))

        # Tokenize the text/text pairs
        features = self.tokenizer.batch_encode_plus(
            text_pairs, max_length=self.max_seq_length, pad_to_max_length=True, truncation=True
        )

        features = {k: [v[i: i + 4] for i in range(0, len(v), 4)] for k, v in features.items()}
        features["label"] = example_batch[self.label_name]

        return features

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset["train"],
                          collate_fn=DataCollatorForMultipleChoice(tokenizer=self.tokenizer, return_tensors="pt"),
                          batch_size=self.train_batch_size, shuffle=True)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["validation"],
                              collate_fn=DataCollatorForMultipleChoice(tokenizer=self.tokenizer, return_tensors="pt"),
                              batch_size=self.eval_batch_size)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x],
                               collate_fn=DataCollatorForMultipleChoice(tokenizer=self.tokenizer, return_tensors="pt"),
                               batch_size=self.eval_batch_size) for x in self.eval_splits]

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["test"],
                              collate_fn=DataCollatorForMultipleChoice(tokenizer=self.tokenizer, return_tensors="pt"),
                              batch_size=self.eval_batch_size)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x],
                               collate_fn=DataCollatorForMultipleChoice(tokenizer=self.tokenizer, return_tensors="pt"),
                               batch_size=self.eval_batch_size) for x in self.eval_splits]
