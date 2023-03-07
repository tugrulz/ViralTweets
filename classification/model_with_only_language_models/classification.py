# MODIFY AS REQUIRED
import torch
import pandas as pd
import seaborn as sns
import numpy as np

import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from datasets import load_dataset
from datasets import Dataset, DatasetDict
from transformers import DataCollatorWithPadding

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from torch.optim import AdamW
from torch.nn import BCEWithLogitsLoss

from transformers import get_scheduler

from tqdm.auto import tqdm

import evaluate

from tqdm import tqdm
import logging
logging.basicConfig(level=logging.INFO)

from text_preprocessing import clean_tweet, clear_reply_mentions, normalizeTweet

'''
DATA_PATH = "../../data"

PROCESSED_PATH = f"{DATA_PATH}/processed"

PROCESSED_PATH_VIRAL = f'{DATA_PATH}/new/processed/viral'
PROCESSED_PATH_COVID = f'{DATA_PATH}/new/processed/covid'
'''

# Different models
BERT_BASE_UNCASED = "bert-base-uncased"
BERT_BASE_CASED = "bert-base-cased"
ROBERTA_BASE = "roberta-base"
BERT_TWEET = "vinai/bertweet-base"

# TODO: Don't forget to cite papers if you use some model
BERT_TINY = "prajjwal1/bert-tiny"

TWEET_MAX_LENGTH = 280

# TEST SPLIT RATIO + MODELS (ADD MORE MODELS FROM ABOVE)
MODELS = [BERT_TWEET, BERT_TINY, BERT_BASE_CASED, ROBERTA_BASE]
TEST_RATIO = 0.2

def preprocess_data(dataset):
    # remove tweets with 0 retweets (to eliminate their effects)
    #dataset = dataset[dataset.retweet_count > 0]

    ## UPDATE: Get tweets tweeted by the same user, on the same day he tweeted a viral tweet

    # Get the date from datetime
    # normalize() sets all datetimes clock to midnight, which is equivalent as keeping only the date part
    dataset['date'] = dataset.created_at.dt.normalize()

    viral_tweets = dataset[dataset.viral]
    non_viral_tweets = dataset[~dataset.viral]

    temp = non_viral_tweets.merge(viral_tweets[['author_id', 'date', 'id', 'viral']], on=['author_id', 'date'], suffixes=(None, '_y'))
    same_day_viral_ids = temp.id_y.unique()

    same_day_viral_tweets = viral_tweets[viral_tweets.id.isin(same_day_viral_ids)].drop_duplicates(subset=['author_id', 'date'])
    same_day_non_viral_tweets = temp.drop_duplicates(subset=['author_id', 'date'])

    logging.info(f"Number of viral tweets tweeted on the same day {len(same_day_viral_tweets)}")
    logging.info(f"Number of non viral tweets tweeted on the same day {len(same_day_non_viral_tweets)}")

    dataset = pd.concat([same_day_viral_tweets, same_day_non_viral_tweets], axis=0)
    dataset = dataset[['id', 'text', 'viral']]

    # Balance classes to have as many viral as non viral ones
    #dataset = pd.concat([positives, negatives.sample(n=len(positives))])
    #dataset = pd.concat([positives.iloc[:100], negatives.sample(n=len(positives)).iloc[:200]])

    # Clean text to prepare for tokenization
    #dataset = dataset.dropna()
    dataset.loc[:, "viral"] = dataset.viral.astype(int)

    # TODO: COMMENT IF YOU WANT TO KEEP TEXT AS IS
    dataset["cleaned_text"] = dataset.text.apply(lambda x: clean_tweet(x, demojize_emojis=False))

    dataset = dataset.dropna()
    dataset = dataset[['id', 'cleaned_text', 'viral']]

    return dataset

def prepare_dataset(sample_data, balance=False):
    # Split the train and test data st each has a fixed proportion of viral tweets
    train_dataset, eval_dataset = train_test_split(sample_data, test_size=TEST_RATIO, random_state=42, stratify=sample_data.viral)

    # Balance test set
    if balance:
        eval_virals = eval_dataset[eval_dataset.viral == 1]
        eval_non_virals = eval_dataset[eval_dataset.viral == 0]
        eval_dataset = pd.concat([eval_virals, eval_non_virals.sample(n=len(eval_virals))])

    logging.info('{:>5,} training samples with {:>5,} positives and {:>5,} negatives'.format(
        len(train_dataset), len(train_dataset[train_dataset.viral == 1]), len(train_dataset[train_dataset.viral == 0])))
    logging.info('{:>5,} validation samples with {:>5,} positives and {:>5,} negatives'.format(
        len(eval_dataset), len(eval_dataset[eval_dataset.viral == 1]), len(eval_dataset[eval_dataset.viral == 0])))

    train_dataset.to_parquet("train.parquet.gzip", compression='gzip')
    eval_dataset.to_parquet("test.parquet.gzip", compression='gzip')

    ds = load_dataset("parquet", data_files={'train': 'train.parquet.gzip', 'test': 'test.parquet.gzip'})
    return ds

def tokenize_function(example, tokenizer):
  # Truncate to max length. Note that a tweet's maximum length is 280
  # TODO: check dynamic padding: https://huggingface.co/course/chapter3/2?fw=pt#dynamic-padding
  return tokenizer(example["cleaned_text"], truncation=True)


def test_all_models(ds, models=MODELS):
    models_losses = {}
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    output = ""

    for checkpoint in models:
        torch.cuda.empty_cache()
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
        model.to(device)

        tokenized_datasets = ds.map(lambda x: tokenize_function(x, tokenizer=tokenizer), batched=True)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        tokenized_datasets = tokenized_datasets.remove_columns(["__index_level_0__", "cleaned_text", "id"])
        tokenized_datasets = tokenized_datasets.rename_column("viral", "labels")
        tokenized_datasets.set_format("torch")

        batch_size = 32

        train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=batch_size, collate_fn=data_collator)
        eval_dataloader = DataLoader(tokenized_datasets["test"], batch_size=batch_size, collate_fn=data_collator)

        criterion = BCEWithLogitsLoss()
        optimizer = AdamW(model.parameters(), lr=5e-5)

        optimizer = AdamW(model.parameters(), lr=5e-5)

        num_epochs = 15
        num_training_steps = num_epochs * len(train_dataloader)
        lr_scheduler = get_scheduler(
            name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
        )

        progress_bar = tqdm(range(num_training_steps))

        losses = []
        model.train()
        for epoch in range(num_epochs):
            for batch in train_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                
                loss = outputs.loss
                losses.append(loss.item())
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

        models_losses[checkpoint] = losses

        metric = evaluate.combine(["accuracy", "recall", "precision", "f1"])
        model.eval()
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])

        output += f"checkpoint: {checkpoint}: {metric.compute()}\n"
    logging.info(output)
    with open("same_day_as_viral_with_features_train_test_balanced_accuracy.txt", "w") as text_file:
        text_file.write(output)
    return models_losses

def main():
    # DATA FILE SHOULD BE AT THE ROOT WITH THIS SCRIPT
    all_tweets_labeled = pd.read_parquet(f'final_dataset_since_october_2022.parquet.gzip')

    dataset = preprocess_data(all_tweets_labeled)
    ds = prepare_dataset(dataset, balance=False)

    test_all_models(ds)

if __name__ == "__main__":
    main()