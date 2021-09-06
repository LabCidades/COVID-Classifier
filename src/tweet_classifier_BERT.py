import argparse
import random
import time
import datetime
import numpy as np
import torch
import pandas as pd
import os
from transformers import AutoModel, AutoModelForPreTraining, AutoTokenizer, pipeline
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

# Taken from: https://mccormickml.com/2019/07/22/BERT-fine-tuning/

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

os.makedirs(os.path.join(os.getcwd(), 'huggingface_cache'), exist_ok=True)

def get_data(file):
    df = pd.read_csv(os.path.join(os.getcwd(),
                                  'annotated_tweets', f"{file}.csv"),
                     usecols=['id', 'tweet', 'label'],
                     na_values=['', '???'],
                     dtype={
        'id': 'str',
        'tweet': 'str',
        'label': 'str'
    })
    df.dropna(inplace=True)
    df['label'] = df['label'].astype(int)

    regex_twitter = r'(@[A-Za-z0-9]+)'
    df['tweet'] = df['tweet'].str.replace(regex_twitter, '', regex=True)
    # label
    # 1 and 3 is signal
    # 0, 2, 4, 5 and 6 is noise
    df['label_binary'] = df['label'].map({
        1: 1,
        3: 1,
        0: 0,
        2: 0,
        4: 0,
        5: 0,
        6: 0
    })
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
            max_length=max_length,           # Pad & truncate all sentences.
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
    labels = torch.tensor(df['label_binary'].values)
    if print_sample:
        # Print sentence 0, now as a list of IDs.
        print('Original: ', df.loc[0, 'tweet'])
        print('Token IDs:', input_ids[0])
    return input_ids, attention_masks, labels


def train_test_split(input_ids, attention_masks, labels, batch_size):
    # Train/Test Split
    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, labels)

    # Create a 90-10 train-test split.

    # Calculate the number of samples to include in each set.
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size

    # Divide the dataset by randomly selecting samples.
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size])

    print(f"{train_size} training samples")
    print(f"{test_size} test samples")

    # DataLoader
    # The DataLoader needs to know our batch size for training, so we specify it
    # here. For fine-tuning BERT on a specific task, the authors recommend a batch
    # size of 16 or 32.
    # My GPU only tolerates 8. 16 or 32 it blows up!
    batch_size = batch_size

    # Create the DataLoaders for our training and test sets.
    # We'll take training samples in random order.
    train_dataloader = DataLoader(
        train_dataset,  # The training samples.
        sampler=RandomSampler(train_dataset),  # Select batches randomly
        batch_size=batch_size  # Trains with this batch size.
    )

    # For test the order doesn't matter, so we'll just read them sequentially.
    test_dataloader = DataLoader(
        test_dataset,  # The test samples.
        # Pull out batches sequentially.
        sampler=SequentialSampler(test_dataset),
        batch_size=batch_size  # Evaluate with this batch size.
    )
    return train_dataloader, test_dataloader


def get_model(cache_dir=os.path.join(os.getcwd(), 'huggingface_cache'), print_model=False):
    # We’ll be using BertForSequenceClassification.
    # This is the normal BERT model with an added single linear layer on
    # top for classification that we will use as a sentence classifier.
    # As we feed input data, the entire pre-trained BERT model and the
    # additional untrained classification layer is trained on our specific
    # task
    # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#bertforsequenceclassification

    # Load BertForSequenceClassification, the pretrained BERT model with a single
    # linear classification layer on top.

    model = BertForSequenceClassification.from_pretrained(
        'neuralmind/bert-base-portuguese-cased',
        # The number of output labels--2 for binary classification.
        num_labels=2,
        # You can increase this for multi-class tasks.
        # Whether the model returns attentions weights.
        output_attentions=False,
        # Whether the model returns all hidden-states.
        output_hidden_states=False,
        return_dict=False,
        cache_dir=cache_dir
    )
    if print_model:
        # Get all of the model's parameters as a list of tuples.
        params = list(model.named_parameters())
        print('The BERT model has {:} different named parameters.\n'.format(
            len(params)))
        print('==== Embedding Layer ====\n')
        for p in params[0:5]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
        print('\n==== First Transformer ====\n')
        for p in params[5:21]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
        print('\n==== Output Layer ====\n')
        for p in params[-4:]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
    return model


def flat_accuracy(preds, labels):
    """Function to calculate the accuracy of our predictions vs labels"""
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def confusion(prediction, truth):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """
    prediction = np.argmax(prediction, axis=1).flatten()
    truth = truth.flatten()
    confusion_vector = prediction / truth
    # Element-wise division of the 2 arrays returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    true_positives = np.sum(confusion_vector == 1)
    false_positives = np.sum(confusion_vector == float('inf'))
    true_negatives = np.sum(np.isnan(confusion_vector))
    false_negatives = np.sum(confusion_vector == 0)

    return true_positives, false_positives, true_negatives, false_negatives


def format_time(elapsed):
    """Takes a time in seconds and returns a string hh:mm:ss"""
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def train_model(model, lr, epochs, train_dataloader, test_dataloader, seed_val=42):
    # For the purposes of fine-tuning, the authors recommend choosing from the
    # following values(from Appendix A.3 of the BERT paper):
    # - Batch size: 16, 32
    # - Learning rate(Adam): 5e-5, 3e-5, 2e-5
    # - Number of epochs: 2, 3, 4
    
    batch_size = test_dataloader.batch_size
    # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
    # I believe the 'W' stands for 'Weight Decay fix"
    optimizer = AdamW(model.parameters(),
                      lr=lr,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                      )

    # Number of training epochs. The BERT authors recommend between 2 and 4.
    # We chose to run for 4, but we'll see later that this may be over-fitting the
    # training data.
    epochs = epochs
    # Total number of training steps is [number of batches] x [number of epochs].
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * epochs
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)

    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

    # Set the seed value all over the place to make this reproducible.
    seed_val = seed_val

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)

    # We'll store a number of quantities such as training and test loss,
    # test accuracy, and timings.
    training_stats = []

    # Measure the total training time for the whole run.
    total_t0 = time.time()

    # For each epoch...
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        model.train()
        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 100 batches.
            if step % 100 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(
                    step, len(train_dataloader), elapsed))

            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because
            # accumulating the gradients is "convenient while training RNNs".
            model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # It returns different numbers of parameters depending on what arguments
            # are given and what flags are set. For our useage here, it returns
            # the loss (because we provided labels) and the "logits"--the model
            # outputs prior to activation.
            loss, logits = model(b_input_ids,
                                 token_type_ids=None,
                                 attention_mask=b_input_mask,
                                 labels=b_labels)

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))

        # ========================================
        #               Test
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our test set.

        print("")
        print("Running Test...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables
        total_test_accuracy = 0
        total_test_loss = 0
        total_test_false_positives = 0
        total_test_false_negatives = 0

        # Evaluate data for one epoch
        for batch in test_dataloader:
            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using
            # the `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():

                # Forward pass, calculate logit predictions.
                # token_type_ids is the same as the "segment ids", which
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                # The documentation for this `model` function is here:
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                # Get the "logits" output by the model. The "logits" are the output
                # values prior to applying an activation function like the softmax.
                loss, logits = model(b_input_ids,
                                     token_type_ids=None,
                                     attention_mask=b_input_mask,
                                     labels=b_labels)

            # Accumulate the test loss.
            total_test_loss += loss.item()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            confusion_matrix = confusion(logits, label_ids)

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_test_accuracy += flat_accuracy(logits, label_ids)
            total_test_false_positives += confusion_matrix[1] / batch_size
            total_test_false_negatives += confusion_matrix[3] / batch_size

        # Report the final accuracy for this test run.
        avg_test_accuracy = total_test_accuracy / len(test_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_test_accuracy))
        # Report the final false positives for this test run.
        avg_test_false_positives = total_test_false_positives / \
            len(test_dataloader)
        print("  False Positives: {0:.2f}".format(avg_test_false_positives))
        # Report the final false negatives for this test run.
        avg_test_false_negatives = total_test_false_negatives / \
            len(test_dataloader)
        print("  False Negatives: {0:.2f}".format(avg_test_false_negatives))

        # Calculate the average loss over all of the batches.
        avg_test_loss = total_test_loss / len(test_dataloader)

        # Measure how long the test run took.
        test_time = format_time(time.time() - t0)

        print("  Test Loss: {0:.2f}".format(avg_test_loss))
        print("  Test took: {:}".format(test_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Test. Loss': avg_test_loss,
                'Test. Accur.': avg_test_accuracy,
                'Test. False Pos.': avg_test_false_positives,
                'Test. False Neg.': avg_test_false_negatives,
                'Training Time': training_time,
                'Test Time': test_time
            }
        )

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(
        format_time(time.time()-total_t0)))

    # Training Stats
    # Create a DataFrame from our training statistics.
    df_stats = pd.DataFrame(data=training_stats)
    # Use the 'epoch' as the row index.
    df_stats = df_stats.set_index('epoch')
    # A hack to force the column headers to wrap.
    # df = df.style.set_table_styles([dict(selector="th",props=[('max-width', '70px')])])
    # Display the table.
    return df_stats


def save_weights(model, fname):
    os.makedirs(os.path.join(os.getcwd(), 'model_weights'), exist_ok=True)
    path = os.path.join(os.getcwd(), 'model_weights', fname)
    torch.save(model.state_dict(), path)


# # Plot the learning curve.
# plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
# plt.plot(df_stats['Test. Loss'], 'g-o', label="Test")
# # Label the plot.
# plt.title("Training & Test Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()
# plt.xticks([1, 2, 3, 4])
# plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Este script treina um classificador de tweets em sinal (1) ou ruído (0) para a presença de sintomas usando um modelo transformer BERT pré-treinado em português BERTimbau', add_help=True)
    parser.add_argument(
        '-f', '--file', help='arquivo de treino com tweets rotulados', type=str)
    parser.add_argument(
        '-lr', '--learning-rate', help='taxa de aprendizagem, do paper do BERT você pode escolher dentre 5e-5, 3e-5 ou 2e-5', default='5e-5', type=float)
    parser.add_argument(
        '-e', '--epoch', help='épocas, do paper do BERT você pode escolher entre 2 a 4', default=3, type=int)
    parser.add_argument(
        '-b', '--batchsize', help='tamanho do batch, ideal ser uma potência de 2, escolha com cuidado para não estourar a memória da GPU', default=8, type=int)
    args = parser.parse_args()

    # Run Stuff
    df = get_data(args.file)
    input_ids, attention_masks, labels = tokenization(df, max_length=250)
    train_dataloader, test_dataloader = train_test_split(
        input_ids=input_ids, attention_masks=attention_masks, labels=labels, batch_size=args.batchsize)
    model = get_model()
    # Tell pytorch to run this model on the CPU or GPU.
    model.to(device)
    results = train_model(model=model, lr=args.learning_rate, epochs=args.epoch,
                          train_dataloader=train_dataloader, test_dataloader=test_dataloader)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    os.makedirs(os.path.join(os.getcwd(), 'results'), exist_ok=True)
    results.to_csv(os.path.join(os.getcwd(), 'results',
                   f"{args.file}-lr_{args.learning_rate}-e_{args.epoch}-batch_{args.batchsize}.csv"))
    save_weights(
        model, fname=f"{args.file}-lr_{args.learning_rate}-e_{args.epoch}-batch_{args.batchsize}.pt")
