import copy
from itertools import chain
from typing import Dict, List
from termcolor import colored
import torch
import transformers
from torch.utils.data import Dataset, IterableDataset
from prompts.utils import format_conversation, tokenize_conversation
from utils.distributed import get_rank, is_main_process
from datasets import load_dataset
from data.utils import load_file_stream
import ast



# datasets for pretrain
def preprocess_pretrain_dataset(
    examples,
    tokenizer,
    cutoff_len=10,
) -> Dict[str, List[List[int]]]:
    # build grouped texts with format `X1 X2 X3 ...` if packing is enabled
    # eos_token = "<|end_of_text|>" if data_args.template == "llama3" else tokenizer.eos_token
    eos_token = tokenizer.eos_token
    # text_examples = [messages[0]["content"] + eos_token for messages in examples["prompt"]]
    text_examples = [e + eos_token for e in examples]

    # if not data_args.packing:
    #     result = tokenizer(text_examples, add_special_tokens=False, max_length=data_args.cutoff_len, truncation=True)
    # else:
    tokenized_examples = tokenizer(text_examples, add_special_tokens=False)
    concatenated_examples = {k: list(chain(*tokenized_examples[k])) for k in tokenized_examples.keys()}
    total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
    block_size = cutoff_len
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }

    return result


class PretrainStreamingDataset(IterableDataset):
    '''
    This class is used to create an iterable streaming dataset for pretrain.
    All text inputs are packed together and split into blocks of `cutoff_len` tokens.
    '''
    def __init__(self, tokenizer, data_name_or_path, cutoff_len=512):
        self.tokenizer = tokenizer
        self.dataset = load_dataset(data_name_or_path, trust_remote_code=True, streaming=True)['train']
        # self.dataset = load_dataset(data_name_or_path, data_dir="python", split="train", streaming=True)
        self.cutoff_len = cutoff_len
        self.has_print = False

    def __iter__(self):
        concatenated_inputs = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        for text in self.dataset:
            text_label = "text" if "text" in text else "content"
            inputs = self.tokenizer(text[text_label] + self.tokenizer.eos_token)
            inputs['labels'] = copy.deepcopy(inputs["input_ids"])
            if is_main_process():
                if not self.has_print:
                    # text = self.tokenizer.decode(inputs["input_ids"], skip_special_tokens=False)
                    # print(text)
                    print_tokens_with_color(inputs["input_ids"], inputs['labels'], self.tokenizer)
                    self.has_print = True
            for k, v in inputs.items():
                concatenated_inputs[k].extend(v)
            total_length = len(concatenated_inputs["input_ids"])

            if total_length > self.cutoff_len:
                for i in range(0, total_length, self.cutoff_len):
                    block = {
                        k: v[i:i+self.cutoff_len] for k, v in concatenated_inputs.items()
                    }
                    if len(block['input_ids']) == self.cutoff_len:
                        yield block
                    else:
                        concatenated_inputs = block

        if len(concatenated_inputs["input_ids"]) > 0:
            yield concatenated_inputs


def process_streamids_inputs(inputs, mask_prefix=False, pivot=-1):
    '''
    Process the inputs for streaming ids dataset. Find -1 in input ids and remove it.
    Set the labels to -100 before -1.
    '''

    # Find -1 in input ids and remove it
    # if -1 in inputs["input_ids"]:
    #     # Find the position of -1
    #     pos = inputs["input_ids"].index(-1)
    #     inputs["input_ids"] = [i for i in inputs["input_ids"] if i != -1]
    #     inputs["attention_mask"] = [1] * len(inputs["input_ids"])
    #     if mask_prefix:
    #         # mask the labels before -1
    #         labels = [-100] * pos + inputs["input_ids"][pos:]
    #         inputs["labels"] = labels
    #     else:
    #         inputs["labels"] = copy.deepcopy(inputs["input_ids"])


    # Replace -1 with 0
    if mask_prefix and pivot in inputs['input_ids']:
        # get position of last pivot
        last_pos = len(inputs["input_ids"]) - 1 - inputs["input_ids"][::-1].index(pivot) + 1
        inputs["labels"] = [-100] * last_pos + inputs["input_ids"][last_pos:]
    else:
        inputs["labels"] = copy.deepcopy(inputs["input_ids"])
    inputs["input_ids"] = [49999 if i < 0 else i for i in inputs["input_ids"]]
    inputs["attention_mask"] = [1] * len(inputs["input_ids"])

    return inputs


def print_tokens_with_color(token_ids, labels, tokenizer):
    for i, label in enumerate(labels):
        token = tokenizer.decode(token_ids[i], skip_special_tokens=False)
        if label == -100:
            print(token, end=' ')
        else:
            print(colored(token, 'blue'), end=' ')

class PretrainStreamingIdsDataset(IterableDataset):
    '''
    This class is used to create an iterable streaming dataset for pretrain.
    All text inputs are packed together and split into blocks of `cutoff_len` tokens.
    '''
    def __init__(self, tokenizer, data_name_or_path, cutoff_len=512, concatenate=True):
        self.tokenizer = tokenizer
        self.concatenate = concatenate
        self.dataset = load_file_stream(data_name_or_path)
        self.cutoff_len = cutoff_len
        self.has_print = False

    def __iter__(self):
        concatenated_inputs = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        for line in self.dataset:
            ids = ast.literal_eval(line[1])
            ids = ids + [self.tokenizer.eos_token_id]
            inputs = {
                "input_ids": ids,
                "attention_mask": [1] * len(ids),
                "labels": copy.deepcopy(ids),
            }
            inputs = process_streamids_inputs(inputs, mask_prefix=True)
            if is_main_process():
                if not self.has_print:
                    # text = self.tokenizer.decode(inputs["input_ids"], skip_special_tokens=False)
                    # print(text)
                    # print(inputs["input_ids"])
                    print_tokens_with_color(inputs["input_ids"], inputs["labels"], self.tokenizer)
                    print(f"Input ids: {inputs['input_ids']}")
                    print(f"Attn mask: {inputs['attention_mask']}")
                    print(f"Labels: {inputs['labels']}")
                    self.has_print = True

            # If concatenate is enabled, split the concatenated inputs into blocks of `cutoff_len` tokens
            if self.concatenate:
                for k, v in inputs.items():
                    concatenated_inputs[k].extend(v)
                total_length = len(concatenated_inputs["input_ids"])
                if total_length > self.cutoff_len:
                    for i in range(0, total_length, self.cutoff_len):
                        block = {
                            k: v[i:i+self.cutoff_len] for k, v in concatenated_inputs.items()
                        }
                        if len(block['input_ids']) == self.cutoff_len:
                            yield block
                        else:
                            concatenated_inputs = block
            else:
                assert len(inputs["input_ids"]) > 0
                yield inputs

        # At the end of the dataset, yield the last block
        if len(concatenated_inputs["input_ids"]) > 0:
            yield concatenated_inputs


class IterableCollator(object):
    '''
    Used for padding, although all inputs has been truncated to `cutoff_len` tokens.
    '''
    def __init__(self, tokenizer, padding_side='right', max_length=512, pad_to_max_length=False):
        self.tokenizer = tokenizer
        self.padding_side = padding_side
        self.tokenizer.pad_token_id = tokenizer.eos_token_id
        self.max_length = max_length
        self.pad_to_max_length = pad_to_max_length

    def __call__(self, instances):
        input_ids = [torch.tensor(e["input_ids"], dtype=torch.long) for e in instances]
        attention_masks = [torch.tensor(e["attention_mask"], dtype=torch.long) for e in instances]
        labels = []
        for e in instances:
            if "labels" in e:
                labels.append(torch.tensor(e["labels"], dtype=torch.long))
            else:
                raise RuntimeError("Labels must be provided for training.")


        if self.padding_side == 'left':
            # pad all inputs from left side, this can help batch generation
            reversed_input_ids = [ids.flip(0) for ids in input_ids]
            reversed_attention_masks = [mask.flip(0) for mask in attention_masks]
            reversed_labels = [label.flip(0) for label in labels]

            padded_input_ids = torch.nn.utils.rnn.pad_sequence(reversed_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            padded_input_ids = padded_input_ids.flip(1)
            padded_attention_masks = torch.nn.utils.rnn.pad_sequence(reversed_attention_masks, batch_first=True, padding_value=0)
            padded_attention_masks = padded_attention_masks.flip(1)
            padded_labels = torch.nn.utils.rnn.pad_sequence(reversed_labels, batch_first=True, padding_value=-100)
            padded_labels = padded_labels.flip(1)
        elif self.padding_side == 'right':
            padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            padded_attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
            padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        else:
            raise RuntimeError("Padding side must 'left' or 'right'.")


        # Pad or truncate the inputs to max_length
        if self.pad_to_max_length:
            length = padded_input_ids.size(1)
            if length > self.max_length:
                padded_input_ids = padded_input_ids[:, :self.max_length]
                padded_attention_masks = padded_attention_masks[:, :self.max_length]
                padded_labels = padded_labels[:, :self.max_length]
            elif length < self.max_length:
                pad_length = self.max_length - length
                padded_input_ids = torch.nn.functional.pad(padded_input_ids, (0, pad_length), value=self.tokenizer.pad_token_id)
                padded_attention_masks = torch.nn.functional.pad(padded_attention_masks, (0, pad_length), value=0)
                padded_labels = torch.nn.functional.pad(padded_labels, (0, pad_length), value=-100)

        assert padded_input_ids.shape[1] == self.max_length, f"Input ids shape: {padded_input_ids.shape}"

        # if is_main_process():
        #     print_tokens_with_color(padded_input_ids[0], padded_labels[0], self.tokenizer)
        #     print(f"Input ids: {padded_input_ids.shape} {padded_input_ids}")
        #     print(f"Attn mask: {padded_attention_masks.shape} {padded_attention_masks}")
        #     print(f"Labels: {padded_labels.shape} {padded_labels}")
            # raise RuntimeError("Stop here")
        # print(f"Input ids: {padded_input_ids.shape} {padded_input_ids}")
        # print(f"Attn mask: {padded_attention_masks.shape} {padded_attention_masks}")
        # print(f"Labels: {padded_labels.shape} {padded_labels}")
        return {"input_ids": padded_input_ids, "attention_mask": padded_attention_masks, "labels": padded_labels}

class PretrainStreamingMixDataset(IterableDataset):
    '''
    Iterable dataset for mixing different types of data.
    '''
    def __init__(self, tokenizer, data_name_or_paths, data_ratios, cutoff_len=512):
        self.tokenizer = tokenizer
        self.datasets = []
        for data_name_or_path in data_name_or_paths:
            # Currently only support csv ids file
            if data_name_or_path.endswith(".csv"):
                self.datasets.append(PretrainStreamingIdsDataset(tokenizer, data_name_or_path, cutoff_len=cutoff_len, concatenate=True))
            else:
                self.datasets.append(PretrainStreamingDataset(tokenizer, data_name_or_path, cutoff_len=cutoff_len))
        assert sum(data_ratios) == 1.0, f"Data ratios {data_ratios} must sum to 1."
        self.data_ratios = data_ratios
        self.has_print = False

    def __iter__(self):
        iterators = [iter(dataset) for dataset in self.datasets]
        # Sampling the dataset based on the ratios
        while True:
            iterator_index = torch.multinomial(torch.tensor(self.data_ratios), 1).item()
            try:
                inputs = next(iterators[iterator_index])
                yield inputs
            except StopIteration:
                print("No more data, reset the iterator.")
                iterators[iterator_index] = iter(self.datasets[iterator_index])


if __name__ == '__main__':
    sources = ['List 5 reasons why learn to code.', 'what is it?']
    targets = ['Improve communication skills.', 'Not like a human.']
    tokenizer = transformers.AutoTokenizer.from_pretrained("cerebras/Cerebras-GPT-590M")
    dataset = CausalLMDataset(tokenizer, sources, targets, max_length=512)
    collator = CausalLMCollator(tokenizer, max_length=512)
    print(dataset[1])
    dataloader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        batch_size=4
    )
    for data in dataloader:
        print(data)

    instructions, responses = load_bias_data()
    dataset = CausalLMDataset(tokenizer, instructions, responses, max_length=512)
    print(dataset[1])
    dataloader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        batch_size=4
    )
    for data in dataloader:
        print(data)
        break




