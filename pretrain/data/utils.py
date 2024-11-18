import json
import random
import csv
import pandas as pd


def read_jsonl_to_list(file_path):
    data_list = []
    with open(file_path, 'r') as file:
        for line in file:
            data_list.append(json.loads(line))
    return data_list


def load_instruction_data(datasets, nums):
    instructions = []
    responses = []
    for (d, n) in zip(datasets, nums):
        data_path = f'../datasets/instruction_dataset/{d}.json'
        with open(data_path, 'r') as f:
            data_list = json.load(f)[:n]

        for sample in data_list:
            instruction = sample['instruction']
            if 'input' in sample:
                instruction = instruction + ' ' + sample['input']
                instruction = instruction.strip()
            response = sample['output']

            instructions.append(instruction)
            responses.append(response)

    return instructions, responses


def load_chat_data(datasets, nums):
    messages_list = []
    for (d, n) in zip(datasets, nums):
        data_path = f'../datasets/chat_dataset/{d}.json'
        with open(data_path, 'r') as f:
            data_list = json.load(f)
        
        # If use less than the total number of conversations
        if n <= len(data_list):
            # randomly sample n conversations
            data_list = random.sample(data_list, n)
        messages_list.extend([data['conversations'] for data in data_list])

    return messages_list


def load_file_stream(file_path):
    with open(file_path, 'r') as file:
        if file_path.endswith('.csv'):
            reader = csv.reader(file, delimiter='\t')
            # Skip header
            next(reader)
            for row in reader:
                yield row
        else:
            for line in file:
                yield line
