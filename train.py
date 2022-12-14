# -*- coding: utf-8 -*-
"""(Personal) Approaching Metaphor with Probabilities -- BERT.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1E_RT3bZMCeFCBfc4J4nzBB52gYQuHic8

# Setup: Simply run these cells and then move to the experiment section below.
"""
from random import shuffle

# import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import RobertaTokenizerFast
from sklearn.metrics import classification_report

from dataset import M_Dataset, MelbertDataset
from model import MetaphorModel, ClassificationCausalModel
from utils import convert_str_indices_to_token_indices

# nltk.download('wordnet')

# from nltk.corpus import wordnet as wn

def train(epochs=5, train_batch_size=32, valid_batch_size=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # print('seed ==', torch.random.initial_seed())
    # seed == 15019865508169630983
    torch.random.manual_seed(15019865508169630983)

    model = ClassificationCausalModel()

    # #######################################################################
    # # train masked classification_model on synsets
    # #######################################################################
    #
    # tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large')
    # all_synsets = list(wn.all_synsets())
    # print(f'Searching {len(all_synsets):,} synsets for non-metaphorical examples.')
    # sentence_mask_indices = dict()
    # for synset in all_synsets:
    #     if not synset.name().endswith('01'):
    #         examples = synset.examples()
    #         if examples:
    #             lemmas = synset.lemma_names()
    #             for example in examples:
    #                 for lemma in lemmas:
    #                     if lemma in example:
    #                         str_start = example.index(lemma)
    #                         str_end = str_start + len(lemma)
    #                         start, end = convert_str_indices_to_token_indices(
    #                             tokenizer, example, [str_start, str_end]
    #                         )
    #                         sentence_mask_indices[example] = [start, end]
    #                         break
    #
    # print(f'Found {len(sentence_mask_indices):,} non-metaphorical synset examples.')
    #
    # optimizer = optim.AdamW(classification_model.parameters(), lr=3e-5, weight_decay=1e-1)
    #
    # sentence_mask_indices = list(sentence_mask_indices.items())
    #
    # for epoch in range(4):
    #     running_loss = 0.
    #     for b in range(0, len(sentence_mask_indices), train_batch_size):
    #         batch = sentence_mask_indices[b:b + train_batch_size]
    #         text = [i[0] for i in batch]
    #         mask_start_end_indices = [i[1] for i in batch]
    #         output, loss = classification_model(text, mask_start_end_indices=mask_start_end_indices)
    #
    #         running_loss += loss.item()
    #
    #         loss.backward()
    #         optimizer.step()
    #         optimizer.zero_grad()
    #
    #         print(f'Epoch {epoch} Batch {b // train_batch_size} Running Loss {running_loss / (b // train_batch_size + 1)}')
    #
    #     shuffle(sentence_mask_indices)

    dataset = MelbertDataset(csv_path='https://www.dropbox.com/s/1j2c13i4wlz647k/train.tsv?dl=1')
    valid_dataset = MelbertDataset(csv_path='https://www.dropbox.com/s/tpqjd5xt8cb3p9e/test.tsv?dl=1')

    # find the weighting of the negative to positive samples to balance loss
    negatives = sum(dataset[target][1] == 0 for target in range(len(dataset)))
    positives = sum(dataset[target][1] == 1 for target in range(len(dataset)))
    pos_weight = negatives / positives
    print('pos_weight is', pos_weight)

    optimizer = optim.AdamW(model.parameters(), lr=1e-10, weight_decay=1e-1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(device))
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, 1e-10, 3e-5, step_size_up=len(dataset) * 2 // train_batch_size,
                                            step_size_down=len(dataset) // train_batch_size, cycle_momentum=False)


    best_state_dict = model.state_dict()
    best_valid_loss = None
    for epoch in range(epochs):
        running_loss = 0.
        for b in range(0, len(dataset), train_batch_size):
            batch = [dataset[b + i] for i in range(0, train_batch_size) if b + i < len(dataset)]
            X = [s[0] for s in batch]
            y = [s[1] for s in batch]
            w_indices = [s[2] for s in batch]
            # max_len = max(len(y_) for y_ in y)
            # y = torch.tensor([(y_ + [0] * max_len)[:max_len] for y_ in y]).float().to(device)

            output = model(X, w_indices)
            loss = criterion(output, torch.tensor(y).to(device).float().reshape(-1, 1))
            running_loss += loss.item()

            print(f'Epoch {epoch} Batch {b} Loss {loss.item()} Running loss: {running_loss / (b // train_batch_size + 1)} LR: {scheduler.get_last_lr()}')

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            scheduler.step()

        # validate
        model.eval()
        y_true = []
        y_pred = []
        val_running_loss = 0.
        for b in range(0, len(valid_dataset), valid_batch_size):
            batch = [valid_dataset[b + i] for i in range(0, valid_batch_size) if b + i < len(valid_dataset)]
            X = [s[0] for s in batch]
            y = [s[1] for s in batch]
            w_indices = [s[2] for s in batch]
            # max_len = max(len(y_) for y_ in y)
            # y_tensor = torch.tensor([(y_ + [0] * max_len)[:max_len] for y_ in y]).float().to(device)
            # for subset in [(y_ + [0] * max_len)[:max_len] for y_ in y]:
            #     for item in subset:
            #         y_true.append(item)
            y_true.extend(y)

            with torch.no_grad():
                output = model(X, w_indices)
                # val_running_loss += criterion(output, y_tensor)
                val_running_loss += criterion(output, torch.tensor(y).to(device).float().reshape(-1, 1))
                output = torch.sigmoid(output)
                y_pred.extend(output.round().flatten().cpu().tolist())
                # for o_ in output.round().flatten().cpu():
                #     y_pred.append(o_)
        print('val_loss ==', val_running_loss / (b // valid_batch_size + 1))
        print(classification_report(np.array(y_true),
                                    np.array(y_pred),
                                    target_names=['no_metaphor', 'metaphor'],
                                    digits=5))

        if (best_valid_loss is None) or (val_running_loss / (b // valid_batch_size + 1) < best_valid_loss):
            best_state_dict = model.state_dict()
            best_valid_loss = val_running_loss / (b // valid_batch_size + 1)
        elif val_running_loss / (b // valid_batch_size + 1) > best_valid_loss:
            model.load_state_dict(best_state_dict)

        # lr_scheduler.step()

        torch.save(model, f'model_{epoch}_{val_running_loss / (b // valid_batch_size + 1)}.pt')

        model.train()

        # shuffle the dataset
        dataset.df = dataset.df.sample(frac=1., random_state=epoch * 1_234_567 + 1)


if __name__ == '__main__':
    train(epochs=5, train_batch_size=32, valid_batch_size=4)
