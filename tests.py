import unittest

from transformers import RobertaTokenizerFast

from utils import convert_str_indices_to_token_indices


class TestStringToTokenFunction(unittest.TestCase):

    def setUp(self):
        self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large')

    def test_double_str_indices_to_token_indices_spaces_first(self):
        # test two sentence, with spaces before and after first
        start, end = convert_str_indices_to_token_indices(self.tokenizer,
                                                          [[' isser a ', 'isser a talk day today, is it not?']],
                                                          [30, 33], two_part=True)
        indices = self.tokenizer([[' isser a ', 'isser a talk day today, is it not?']])['input_ids'][0][start:end + 1]
        decoded = self.tokenizer.decode(indices)
        self.assertEqual(decoded, ' not', (f'tokenizer decoded \'{decoded}\' from token indices {indices} '
                                           f'with start and end indices {(start, end)} != \' not\''))

    def test_double_str_indices_to_token_indices_spaces_each(self):
        # test two sentence, with spaces before and after each
        start, end = convert_str_indices_to_token_indices(self.tokenizer,
                                                          [[' isser a ', ' isser a talk day today, is it not? ']],
                                                          [31, 34], two_part=True)
        indices = self.tokenizer([[' isser a ', 'isser a talk day today, is it not? ']])['input_ids'][0][start:end + 1]
        decoded = self.tokenizer.decode(indices)
        self.assertEqual(decoded, ' not', (f'tokenizer decoded \'{decoded}\' from token indices {indices} '
                                           f'with start and end indices {(start, end)} != \' not\''))

    def test_double_str_indices_to_token_indices_spaces_second(self):
        # test two sentence, with spaces before and after second
        start, end = convert_str_indices_to_token_indices(self.tokenizer,
                                                          [['isser a', 'isser a talk day today, is it not? '.strip()]],
                                                          [30, 33], two_part=True)
        indices = self.tokenizer([['isser a', 'isser a talk day today, is it not? '.strip()]])['input_ids'][0][
                  start:end + 1]
        decoded = self.tokenizer.decode(indices)
        self.assertEqual(decoded, ' not', (f'tokenizer decoded \'{decoded}\' from token indices {indices} '
                                           f'with start and end indices {(start, end)} != \' not\''))

    def test_single_str_indices_to_token_indices_no_spaces(self):
        # test one sentence, with no spaces before and after
        start, end = convert_str_indices_to_token_indices(self.tokenizer,
                                                          'isser a',
                                                          [0, 7], two_part=False)
        indices = self.tokenizer('isser a')['input_ids'][start:end + 1]
        decoded = self.tokenizer.decode(indices)
        self.assertEqual(decoded, 'isser a', (f'tokenizer decoded \'{decoded}\' from token indices {indices} '
                                              f'with start and end indices {(start, end)} != \'isser a\''))

    def test_single_str_indices_to_token_indices_spaces(self):
        # test one sentence, with spaces before and after
        start, end = convert_str_indices_to_token_indices(self.tokenizer,
                                                          ' isser a ',
                                                          [0, 8], two_part=False)
        indices = self.tokenizer(' isser a ')['input_ids'][start:end + 1]
        decoded = self.tokenizer.decode(indices)
        self.assertEqual(decoded, ' isser a', (f'tokenizer decoded \'{decoded}\' from token indices {indices} '
                                               f'with start and end indices {(start, end)} != \'isser a\''))


if __name__ == '__main__':
    unittest.main()
