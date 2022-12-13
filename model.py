import re
from collections import defaultdict

import torch
import torch.nn as nn
from transformers import (BertForMaskedLM, BertForTokenClassification, BertTokenizerFast,
                          RobertaForMaskedLM, RobertaForTokenClassification, RobertaTokenizerFast,
                          RobertaModel, RobertaForSequenceClassification, RobertaForCausalLM)

from utils import convert_str_indices_to_token_indices


class ClassificationCausalModel(nn.Module):

    def __init__(self):
        super(ClassificationCausalModel, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large')
        self.classification_model = RobertaForTokenClassification.from_pretrained('roberta-large', num_labels=1,
                                                                                  hidden_dropout_prob=0.2,
                                                                                  attention_probs_dropout_prob=0.2)
        self.causal_model = RobertaForCausalLM.from_pretrained('roberta-large',
                                                               hidden_dropout_prob=0.2,
                                                               attention_probs_dropout_prob=0.2)
        self.post_causal_layer1 = nn.Linear(len(self.tokenizer), len(self.tokenizer) // 64)
        self.post_causal_layer2 = nn.Linear(len(self.tokenizer) // 64, len(self.tokenizer) // 256)
        self.post_causal_layer3 = nn.Linear(len(self.tokenizer) // 256, 1)
        self.final_layer = nn.Linear(4, 1)
        self.to(self.device)

    def forward(self, sentences, word_indices):
        """
        Run sentences through the forward method.

        Parameters
        ----------
        sentences : list of str
            The full sentences to search for metaphors (including the
            word in question).
        word_indices : list of int
            A list comprised of a single word (not string or token!)
            index per sentence.

        Returns
        -------
        Raw tensor of predictions of metaphor [0-1] with shape (batch, 1).
        """

        samples = []

        ###############################################################
        # CONVERT ALL STRING INDICES TO TOKEN INDICES (for this task)
        ###############################################################

        inputs_for_finial_layer, double_tokens, single_tokens, single_sentence_tokens = [], [], [], []
        for i, (sentence, word_index) in enumerate(zip(sentences, word_indices)):
            word_matches = list(re.finditer(r'[^\s]+(?=\s+|$)', sentence))
            str_start, str_end = [word_matches[word_index].start(), word_matches[word_index].end()]
            samples.append([' ' + sentence[str_start:str_end] + ' ', sentence])
            double_token_start, double_token_end = convert_str_indices_to_token_indices(self.tokenizer,
                                                                                        [[' ' + sentence[
                                                                                                str_start:str_end] + ' ',
                                                                                          sentence]],
                                                                                        [str_start, str_end],
                                                                                        two_part=True)
            double_tokens.append([double_token_start, double_token_end])
            single_token_start, single_token_end = convert_str_indices_to_token_indices(self.tokenizer,
                                                                                        ' ' + sentence[
                                                                                              str_start:str_end] + ' ',
                                                                                        [0, len(sentence[
                                                                                                str_start:str_end]) + 1],
                                                                                        # essentially +2 for spaces
                                                                                        two_part=False)
            single_tokens.append([single_token_start, single_token_end])

            single_sentence_token_start, single_sentence_token_end = convert_str_indices_to_token_indices(
                self.tokenizer,
                sentence,
                [str_start, str_end],
                two_part=False)
            single_sentence_tokens.append([single_sentence_token_start, single_sentence_token_end])

        ###############################################################
        # GET RELEVANT PORTIONS OF CLASSIFICATION LOGITS
        ###############################################################

        tokenized = self.tokenizer(samples,
                                   return_tensors='pt',
                                   padding=True).to(self.device)

        # set position ids for the tokens in question to 0
        tokenized['position_ids'] = torch.arange(tokenized['input_ids'].shape[1]).repeat(
            tokenized['input_ids'].shape[0], 1).to(self.device)
        for i, ((double_token_start, double_token_end), (single_token_start, single_token_end)) in enumerate(zip(double_tokens, single_tokens)):
            tokenized['position_ids'][i][single_token_start:single_token_end + 1] = 0
            tokenized['position_ids'][i][double_token_start:double_token_end + 1] = 0

        # collect outputs from each index of the tokens in question and the [CLS] index
        outputs = self.classification_model(**tokenized)
        logits = outputs.logits.squeeze(-1)
        for i, ((double_token_start, double_token_end), (single_token_start, single_token_end)) in enumerate(zip(double_tokens, single_tokens)):
            output1 = logits[i][double_token_start:double_token_end + 1].mean().reshape(1)
            output2 = logits[i][single_token_start:single_token_end + 1].mean().reshape(1)
            output3 = logits[i][0].reshape(1)
            inputs_for_finial_layer.append(torch.cat([output1, output2, output3], dim=0))
            if torch.all(torch.isnan(output1)):
                print(samples[i], single_token_start, single_token_end, double_token_start, double_token_end)
                raise Exception('found an nan!')

        ###############################################################
        # GET RELEVANT CAUSAL LM LOGITS
        ###############################################################

        tokenized_for_mask = self.tokenizer([s[1] for s in samples],
                                            return_tensors='pt',
                                            padding=True).to(self.device)

        # set tokens in question to the mask token id
        for i, (single_sentence_token_start, single_sentence_token_end) in enumerate(single_sentence_tokens):
            tokenized_for_mask['input_ids'][i][single_sentence_token_start:single_sentence_token_end + 1] = self.tokenizer.mask_token_id

        # get the causal lm outputs, and aggregate the masked indices into a single tensor (1, vocab_size)
        masked_outputs = self.causal_model(**tokenized_for_mask)
        logits = masked_outputs.logits
        masked_inputs = []
        for i, (single_sentence_token_start, single_sentence_token_end) in enumerate(single_sentence_tokens):
            masked_output = logits[i][single_sentence_token_start:single_sentence_token_end + 1]
            masked_mean = torch.mean(masked_output, dim=0)
            masked_inputs.append(masked_mean)

        # run the causal outputs for the relevant indices through the post causal layers
        post_causal_layer1_outputs = nn.functional.leaky_relu(
            self.post_causal_layer1(torch.stack(masked_inputs, dim=0)), negative_slope=0.2)
        post_causal_layer2_outputs = nn.functional.leaky_relu(
            self.post_causal_layer2(post_causal_layer1_outputs), negative_slope=0.2)
        post_causal_layer3_outputs = nn.functional.leaky_relu(self.post_causal_layer3(post_causal_layer2_outputs), negative_slope=0.2)

        ###############################################################
        # CONCATENATE ALL 4 OUTPUTS AND RUN THROUGH FINAL LAYER
        ###############################################################

        first_three_final_inputs = torch.stack(inputs_for_finial_layer).to(self.device)
        inputs_for_finial_layer = torch.cat([first_three_final_inputs, post_causal_layer3_outputs], dim=1)

        return self.final_layer(inputs_for_finial_layer)


class MetaphorSimilarityModel(nn.Module):

    def __init__(self):
        super(MetaphorSimilarityModel, self).__init__()
        self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.roberta = RobertaModel.from_pretrained('roberta-large')
        # self.roberta.embeddings.position_embeddings.weight.data = torch.mean(
        #     self.roberta.embeddings.position_embeddings.weight.data, dim=0).unsqueeze(0).repeat(514, 1)
        self.embeddings = torch.zeros(1, 1024).to(self.device)
        self.embedding_labels = torch.zeros(1).to(self.device)
        self.token_indices = defaultdict(list)
        self.to(self.device)

    def create_embeddings(self, training_dataset):
        with torch.no_grad():
            for b in range(0, len(training_dataset)):
                X = training_dataset[b][0]
                tokenized_X = self.tokenizer(
                    X, return_tensors='pt', padding=True
                ).to(self.device)
                y = training_dataset[b][1]

                output = self.roberta(**tokenized_X)
                for embedding, label, input_id in zip(output.last_hidden_state[0], y, tokenized_X['input_ids'][0]):
                    self.token_indices[input_id.item()].append(len(self.embeddings))
                    self.embeddings = torch.cat([self.embeddings, embedding.unsqueeze(0)], dim=0)
                    self.embedding_labels = torch.cat([self.embedding_labels, torch.tensor([label]).to(self.device)],
                                                      dim=0)
        print('embedding shape is', self.embeddings.shape)

    def forward(self, sentence, k=5):
        with torch.no_grad():
            tokenized_X = self.tokenizer(
                sentence, return_tensors='pt', padding=True
            ).to(self.device)
            output = self.roberta(**tokenized_X)
            labels = []
            for hidden_state, input_id in zip(output.last_hidden_state[0], tokenized_X['input_ids'][0]):
                sub_embeddings = self.embeddings[self.token_indices.get(input_id.item(), [0])]
                sub_labels = self.embedding_labels[self.token_indices.get(input_id.item(), [0])]
                similarities = torch.cosine_similarity(hidden_state.unsqueeze(0), sub_embeddings)
                topk = torch.topk(similarities, k=k)
                labels.append(torch.mean(sub_labels[topk.indices]).round())
        return labels


class MetaphorModelFixed(nn.Module):

    def __init__(self,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(MetaphorModel, self).__init__()
        self.final = nn.Linear(in_features=2, out_features=1, bias=True)
        self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large')
        self.prefix_model = RobertaForSequenceClassification.from_pretrained('roberta-large', num_labels=1)
        self.postfix_model = RobertaForSequenceClassification.from_pretrained('roberta-large', num_labels=1)
        self.postfix_model.roberta.embeddings = self.prefix_model.roberta.embeddings
        self.postfix_model.roberta.encoder = self.prefix_model.roberta.encoder
        self.device = device
        self.to(device)

    def forward(self, texts, mask_indices):
        for mask_index in mask_indices:
            pass  # TODO: finish implementation
        prefix_tokens = self.tokenizer(list(zip(texts, tokens)), return_tensors='pt', padding=True).to(self.device)
        prefix_outputs = self.prefix_model(**prefix_tokens)
        postfix_tokens = self.tokenizer(list(zip(tokens, texts)), return_tensors='pt', padding=True).to(self.device)
        postfix_outputs = self.postfix_model(**postfix_tokens)

        x = torch.cat([prefix_outputs.logits, postfix_outputs.logits], dim=-1)
        return self.final(x)


class MetaphorModel(nn.Module):

    def __init__(self,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(MetaphorModel, self).__init__()
        self.final = nn.Linear(in_features=2, out_features=1, bias=True)
        self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large')
        self.model_for_mask = RobertaForMaskedLM.from_pretrained('roberta-large')
        self.model_for_tokens = RobertaForTokenClassification.from_pretrained('roberta-large', num_labels=1)
        self.model_for_mask.roberta.embeddings.position_embeddings.weight.data = torch.mean(
            self.model_for_mask.roberta.embeddings.position_embeddings.weight.data, dim=0).unsqueeze(0).repeat(514, 1)
        # self.model_base = RobertaModel.from_pretrained('roberta-large')
        # self.model_for_mask.roberta.embeddings = self.model_base.embeddings
        # self.model_for_mask.roberta.encoder = self.model_base.encoder
        self.model_for_tokens.roberta.embeddings = self.model_for_mask.roberta.embeddings
        self.model_for_tokens.roberta.encoder = self.model_for_mask.roberta.encoder
        self.device = device
        self.to(device)

    def forward(self, text, mask_start_end_indices=[]):
        tokens = self.tokenizer(text, return_tensors='pt', padding=True).to(self.device)
        if mask_start_end_indices:
            labels = []
            for i, (start, end) in enumerate(mask_start_end_indices):
                labels.append(torch.clone(tokens['input_ids'][i]).to(self.device))
                labels[-1][:start] = -100
                labels[-1][end + 1:] = -100
                tokens['input_ids'][i, start:end + 1] = self.tokenizer.mask_token_id
            labels = torch.stack(labels, dim=0)
            outputs = self.model_for_mask(**tokens, labels=labels)
            return outputs.logits, outputs.loss

        probs = torch.empty_like(tokens['input_ids'])
        for i in range(tokens['input_ids'].shape[-1]):
            cloned_input_ids = torch.clone(tokens['input_ids'])
            these_input_ids = torch.clone(cloned_input_ids[:, i])
            cloned_input_ids[:, i] = self.tokenizer.mask_token_id
            output = self.model_for_mask(
                input_ids=cloned_input_ids, attention_mask=tokens['attention_mask']
            ).logits
            preds = output[list(range(probs.shape[0])), i, these_input_ids]
            probs[:, i] = preds
        token_x = self.model_for_tokens(**tokens)  # need last hidden state (b, n, 2) instead of pooler (b, 2)
        x = torch.cat([probs.unsqueeze(-1), token_x.logits], dim=-1)
        return self.final(x).squeeze(-1)
