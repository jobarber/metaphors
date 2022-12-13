import pandas as pd
from torch.utils.data import Dataset

from utils import convert_str_indices_to_token_indices


class MelbertDataset(Dataset):

    def __init__(self, csv_path='https://www.dropbox.com/s/1j2c13i4wlz647k/train.tsv?dl=1'):
        self.df = pd.read_csv(csv_path, sep='\t' if 'tsv' in csv_path else ',', quotechar='`', quoting=3, doublequote=False)
        self.df = self.df.sample(frac=1., random_state=1234)
        print(self.df)

    def __getitem__(self, item):
        row = self.df.iloc[item]
        return row['sentence'], row['label'], row['w_index'], row['label']

    def __len__(self):
        return len(self.df)


class M_Dataset(Dataset):

    def __init__(self, tokenizer,
                 csv_path='https://www.dropbox.com/s/tpqjd5xt8cb3p9e/test.tsv?dl=1',
                 train=True):
        self.df = pd.read_csv(csv_path, sep='\t' if 'tsv' in csv_path else ',')
        if 'sentence_txt' in self.df.columns:
            self.df = self.df.dropna(subset=['sentence_txt'])
        self.df = self.df.sample(frac=1., random_state=1234)
        if train != -1:
            self.df = self.df.iloc[:int(len(self.df) * 0.7)] if train else self.df.iloc[int(len(self.df) * 0.7):]
        self.tokenizer = tokenizer
        self.target_marker = self.tokenizer.encode('M_', add_special_tokens=False)
        self._archived_items = dict()

    def __getitem__(self, item):
        if item in self._archived_items:
            return self._archived_items[item]
        if 'sentence_txt' in self.df.columns:
            text = self.df.iloc[item]['sentence_txt']
        else:
            text = self.df.iloc[item]['sentence']
        M, _ = False, False
        indices = []
        final_chars = ''
        start, end = None, None
        for i, char in enumerate(text):
            if text[i:i + 2] == 'M_':
                M = True
            elif M and char == '_':
                _ = True
            elif M and _ and start is None:
                start = len(final_chars)
                final_chars += char
            elif start is not None and char in ' .,:;':
                end = len(final_chars)
                indices.append((start, end))
                start, end = None, None
                M, _ = False, False
                final_chars += char
            else:
                final_chars += char

        input_ids = self.tokenizer.encode(final_chars)
        target = [0 for input_id in input_ids]
        for start_end in indices:
            start, end = convert_str_indices_to_token_indices(self.tokenizer,
                                                              final_chars,
                                                              start_end)
            target[start:end + 1] = [1 for i in range(start, end + 1)]

        # print(text)
        # pprint(list(zip([self.tokenizer.decode([input_id]) for input_id in input_ids], target)))

        if 'label' in self.df.columns:
            w_index = self.df.iloc[item]['w_index']
            label = self.df.iloc[item]['label']
            # self._archived_items[item] = (final_chars, target, w_index, label)
            return final_chars, target, w_index, label
        # self._archived_items[item] = (final_chars, target)
        return final_chars, target

    def __len__(self):
        return len(self.df)
