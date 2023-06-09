from data_loader import TweetEvalDataModule
from transformers import AutoModel
import torch
from tqdm import tqdm
import argparse


@torch.no_grad()
def extract_features(model, device, batch):
    input_ids = batch['input_ids'].to(device).squeeze()
    attention_mask = batch['attention_mask'].to(device).squeeze()
    labels = batch['label']

    # calculates hidden states with frozen Transformer model
    outputs = model(input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=None,
                    output_hidden_states=None,
                    return_dict=None
                    )

    hidden_states = outputs[0]
    pooled_output = hidden_states[:, 0]
    return pooled_output.detach().cpu(), labels


def main(args):
    # Configuration
    device = args.device
    batch_size = args.batch_size
    model_name_or_path = args.model_name_or_path

    # Load SWAG DataModule and prepare data
    dm = TweetEvalDataModule(model_name_or_path, "irony",max_seq_length=30,  train_batch_size=batch_size, eval_batch_size=batch_size)
    dm.prepare_data()  # downloads data if needed
    dm.setup("fit")  # splits and preprocesses data
    train_dl, valid_dl = dm.train_dataloader(), dm.val_dataloader()

    # Load model
    model = AutoModel.from_pretrained(model_name_or_path).to(device)
    model.eval()

    # Extract features
    for split, dl in zip(["train", "valid"], [train_dl, valid_dl]):
        all_features, all_labels = [], []
        for batch in tqdm(dl):
            features, labels = extract_features(model, device, batch)
            all_features.append(features)
            all_labels.append(labels)

        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        torch.save(all_features, "tweet_eval_{}_features.pt".format(split))
        torch.save(all_labels, "tweet_eval_{}_labels.pt".format(split))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Lab2 - Feature extraction')
    # Configuration
    parser.add_argument('--batch_size', type=int,
                        default=128, help='input batch size')
    parser.add_argument('--device', type=str, default='cuda',
                        help='ID of GPUs to use, eg. cuda:0,cuda:1')
    parser.add_argument('--model_name_or_path', type=str, default='distilbert-base-uncased',
                        help='Path to a pretrained model or model identifier from huggingface.co/models')
    args = parser.parse_args()

    main(args)
