import argparse
import datetime
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AutoModel, AutoModelForPreTraining, AutoTokenizer, pipeline
from transformers import AdamW, BertConfig, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup

from tweet_classifier_BERT import format_time, get_model

# If there's a GPU available...
if torch.cuda.is_available():
    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


def get_data(year):
    df = pd.read_csv(os.path.join(os.getcwd(),
                                  'data', f"twitter_raw_{year}.csv"),
                     usecols=['id', 'date', 'tweet'],
                     dtype={
                            'id': 'str',
                            'tweet': 'str',
                     },
                     parse_dates=['date'])
    regex_twitter = r'(@[A-Za-z0-9]+)'
    df['tweet'] = df['tweet'].str.replace(regex_twitter, '', regex=True)
    return df


def tokenization(df, print_sample=False, max_length=250, cache_dir=os.path.join(os.getcwd(), 'huggingface_cache')):
    # Tokenization
    # Custom BERTimbau here: https://huggingface.co/neuralmind/bert-base-portuguese-cased
    tokenizer = AutoTokenizer.from_pretrained(
        'neuralmind/bert-base-portuguese-cased', do_lower_case=False, cache_dir=cache_dir)
    # Max length
    max_len = 0
    # For every sentence...
    for sent in df['tweet']:
        # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
        input_ids = tokenizer.encode(sent, add_special_tokens=True)
        # Update the maximum sentence length.
        max_len = max(max_len, len(input_ids))
    # 163 so let's use 250 as a precaution
    print('Max sentence length: ', max_len)

    # tokenize all of the sentences and map the tokens to their word IDs.
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in df['tweet']:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
            sent,                      # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            # Pad & truncate all sentences.
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,   # Construct attn. masks.
            return_tensors='pt',     # Return pytorch tensors.
        )
        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    if print_sample:
        # Print sentence 0, now as a list of IDs.
        print('Original: ', df.loc[0, 'tweet'])
        print('Token IDs:', input_ids[0])
    return input_ids, attention_masks


def get_loader(input_ids, attention_masks, batch_size):
    prediction_data = TensorDataset(input_ids, attention_masks)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(
        prediction_data, sampler=prediction_sampler, batch_size=batch_size)
    return prediction_dataloader


def get_predictions(df, model, batch_size):
    # Measure how long the prediction takes.
    t0 = time.time()

    input_ids, attention_masks = tokenization(df, max_length=250)
    prediction_dataloader = get_loader(input_ids, attention_masks, batch_size)

    predictions = []
    # Predict
    for step, batch in enumerate(prediction_dataloader):
        # Progress update every 10000 batches.
        if step % 10000 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)

            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(
                step, len(prediction_dataloader), elapsed))
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask)
        logits = outputs[0]
        # Move logits to CPU
        logits = logits.detach().cpu().numpy()
        # Store predictions and true labels
        predict = np.argmax(logits, axis=1)
        predictions.append(predict)

    # Measure how long this prediction took.
    prediction_time = format_time(time.time() - t0)
    print("")
    print("  Prediction took: {:}".format(prediction_time))
    # Flatten the list
    predictions = [val for sublist in predictions for val in sublist]
    return predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Este script usa o modelo treinado na etapa anterior e classifica os tweets em sinal (1) ou ruído (0) para a presença de sintomas', add_help=True)
    parser.add_argument(
        '-f', '--file', help='arquivo de treino com tweets rotulados', type=str)
    parser.add_argument(
        '-lr', '--learning-rate', help='taxa de aprendizagem do modelo que você deseja usar', type=float)
    parser.add_argument(
        '-e', '--epoch', help='épocas do modelo que você deseja usar', type=int)
    parser.add_argument(
        '-b', '--batchsize', help='tamanho do batch do modelo que você deseja usar', type=int)
    args = parser.parse_args()

    # Prepare predictions/ dir
    os.makedirs(os.path.join(os.getcwd(), 'predictions'), exist_ok=True)

    # Make Prediction
    model = get_model()
    model_weights = os.path.join(os.getcwd(), 'model_weights',
                                 f"{args.file}-lr_{args.learning_rate}-e_{args.epoch}-batch_{args.batchsize}.pt")
    model.load_state_dict(torch.load(model_weights))
    # Model is in evaluation mode
    model.eval()
    # Tell pytorch to run this model on the CPU or GPU.
    model.to(device)

    # Run Stuff
    for year in [2019, 2020, 2021]:
        print(f"Predicting year: {year}")

        df = get_data(year)
        df['label'] = get_predictions(df, model, batch_size=args.batchsize)
        df.to_csv(os.path.join(os.getcwd(), 'predictions',
                  f"{args.file}-twitter_pred_{year}.csv"))

        # Clean cache for next pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
