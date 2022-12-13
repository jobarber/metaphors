import torch
from transformers import BertTokenizerFast, RobertaTokenizerFast


def convert_str_indices_to_token_indices(fast_tokenizer,
                                         text,
                                         start_end_str_indices,
                                         two_part=False,
                                         **tokenizer_kwargs):
    """
    Converts string indices common in QA tasks to token indices.
    Parameters
    ----------
    fast_tokenizer : instance of PreTrainedTokenizerFast
        We have to use a fast tokenizer in order to access offset mappings.
    text : str or list of (single) list of 2 strings (for
        2 sent tasks)
        The full text of the question and context or just the context,
        depending on the situation.
    start_end_str_indices : sequence with two integers.
        Contains start string index and end string index of the answer.
    tokenizer_kwargs : dict
        Any remaining keyword arguments.
    Example
    -------
    # >>> tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    # >>> question_context = [['How many toys are on the floor?',
    #                          'The floor was covered in toys. 15 of them to be exact.']]
    # >>> convert_str_indices_to_token_indices(tokenizer,
    #                                          question_context,
    #                                          [62, 64])
    (16, 18)

    Returns
    -------
    A tuple with the start token index and end token index.
    """
    tokenized = fast_tokenizer(text,
                               return_offsets_mapping=True,
                               return_tensors='pt',
                               **tokenizer_kwargs)

    offset_mapping = tokenized['offset_mapping'][0]

    span = [0, 0]
    # offset_add, last_offset = 0, 0
    search_active = not two_part
    for i, offset in enumerate(offset_mapping):
        # print(start_end_str_indices, offset + offset_add, offset)
        if i > 0 and torch.equal(offset, torch.tensor([0, 0])):
            # offset_add += last_offset
            search_active = True
        elif torch.equal(offset, torch.tensor([0, 0])):
            continue
        if search_active and offset[0] <= start_end_str_indices[0] + 1 <= offset[1]:  # add 1 for the beginning token
            span[0] = i
        if search_active and start_end_str_indices[1] == offset[1]:
            span[1] = i
        if search_active and start_end_str_indices[1] < offset[0]:
            break
        # if offset[-1] != 0:
        #     last_offset = offset[-1]

    return span[0], span[-1]
