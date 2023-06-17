from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from typing import Optional, Union


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features):
        label_name = "label"

        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        features = [{
            "input_ids": f["input_ids"],
            "attention_mask": f["attention_mask"]
        } for f in features]

        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch[label_name] = torch.tensor(labels, dtype=torch.int64)
        return batch
