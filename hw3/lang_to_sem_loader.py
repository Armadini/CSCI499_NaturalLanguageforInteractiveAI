import json
from black import out
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils import preprocess_string, build_output_tables, build_tokenizer_table
from functools import reduce

DATA_FILEPATH = "lang_to_sem_data.json"


class LangToSemDataset(Dataset):

    def __init__(self, input_path: str, is_train: bool, vocab_size: int = 1000, debug=False, episode_len=24, join_instructions=True):
        """Initializes a lang_to_sem_data dataset from JSON file."""
        self.vocab_size = vocab_size

        # Open the dataset file
        with open(input_path) as jsf:
            raw_data = json.load(jsf)

        # Read train data and build tables
        train_data = raw_data["train"]
        self.build_tables(train_data)
        # self.seq_len = 55

        # Use train or valid
        data_ = train_data if is_train else raw_data["valid_seen"]
        print(sum([len(x) for x in train_data]) / len(train_data))
        data = self.clean(data_, episode_len)
        # data = list(map(self.strings_to_number_tensors, data))

        if debug:
            data = data[:3]

        actions_objects = [
            torch.stack([ao for i, ao in sl]) for sl in data]
        print("AO", actions_objects)
        print(len(actions_objects))
        print(len(actions_objects[0]))
        print(len(actions_objects[0][0]))
        # actions_objects = [torch.LongTensor([torch.LongTensor(x) for x in sl]) for sl in actions_objects]

        if join_instructions:
            inputs = [[torch.LongTensor(i) for i, ao in sl] for sl in data]
            inputs = [torch.cat(t) for t in inputs]
        else:
            inputs = [torch.LongTensor([i for i, ao in sl]) for sl in data]

        # inputs = [torch.LongTensor(reduce(lambda a,b:a+[self.vocab_to_index["<sep>"],]+b,sl)) for sl in inputs] 
        self.inputs, self.actions_objects = torch.stack(
            inputs), torch.stack(actions_objects)
        # inputs = [torch.LongTensor(reduce(lambda a,b:torch.cat((a,b)),sl)) for sl in self.inputs]
        print("i a o size", len(self.inputs), len(self.actions_objects))
        print(self.inputs[0])
        print(self.actions_objects[0])

        self.n_data = self.inputs.size(0)

    def build_tables(self, data):
        output_tables, tokenizer_tables = build_output_tables(
            data), build_tokenizer_table(data, vocab_size=self.vocab_size)
        self.actions_to_index, self.index_to_actions = output_tables[:2]
        self.targets_to_index, self.index_to_targets = output_tables[2:]
        self.vocab_to_index, self.index_to_vocab, self.seq_len = tokenizer_tables

        self.max_actions, self.max_objects = len(
            self.actions_to_index), len(self.targets_to_index)

    def clean(self, data, episode_len):
        # Flatten into one long list of shape: (instruction, (action, object))
        DEFAULT = (torch.LongTensor([self.vocab_to_index["<pad>"]] * self.seq_len), torch.LongTensor(([1] + [0] * (self.max_actions-1)) + ([1] + [0] * (self.max_objects-1))))
        # print(DEFAULT)
        return [[self.strings_to_number_tensors((preprocess_string(instruction), a_o)) for (instruction, a_o) in sublist[:min(episode_len, len(sublist))]] + [DEFAULT for _ in range(max(episode_len - len(sublist), 0))]
                for sublist in data]


    def strings_to_number_tensors(self, datapoint):
        instruction, (action, object) = datapoint

        instruction_nums = [self.vocab_to_index[word]
                            for word in instruction.split(" ") if word in self.vocab_to_index]
        if len(instruction_nums) > self.seq_len-2:
            instruction_nums = instruction_nums[:self.seq_len-2]
        instruction_nums = [self.vocab_to_index["<start>"]] + instruction_nums + [self.vocab_to_index["<end>"]]
        instruction_nums = instruction_nums + \
            [self.vocab_to_index["<pad>"]] * \
            (self.seq_len - len(instruction_nums))

        action_num, object_num = self.actions_to_index[action], self.targets_to_index[object]
        action_vec = [int(i == action_num) for i in range(self.max_actions)]
        object_vec = [int(i == object_num) for i in range(self.max_objects)]

        return [torch.LongTensor(instruction_nums), torch.LongTensor(action_vec + object_vec)]

    def __getitem__(self, index):
        # Get an index of the dataset
        return torch.LongTensor(self.inputs[index]), torch.LongTensor(self.actions_objects[index])

    def __len__(self):
        # Number of instances
        return self.n_data


def get_loaders(input_path, batch_size=4, shuffle=False, debug=False, join_instructions=True):
    train_dataset = LangToSemDataset(
        input_path=input_path, is_train=True, debug=debug, join_instructions=join_instructions)
    valid_dataset = LangToSemDataset(
        input_path=input_path, is_train=False, debug=debug, join_instructions=join_instructions)

    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=shuffle)
    valid_dataloader = DataLoader(
        dataset=valid_dataset, batch_size=batch_size, shuffle=shuffle)

    metadata = {"max_actions": train_dataset.max_actions,
                "max_objects": train_dataset.max_objects,
                "seq_len": train_dataset.seq_len}

    return train_dataloader, valid_dataloader, metadata
