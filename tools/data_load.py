import json
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
from datasets import load_dataset

from tools.chat_template import few_shot_template


class ListDataset(Dataset):
    def __init__(self, data: list[str]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


def custom_collate_fn_for_unlabeled_data(batch):
    return batch


def custom_collate_fn_for_labeled_data(batch):
    data_points, labels = zip(*batch)
    labels = default_collate(labels)
    return data_points, labels


def msg_add(prompt: str, **kwargs) -> list[dict]:
    """ Change the prompts to message format. """
    message = []

    if kwargs.get("system_prompt", None) is not None:
        message.append({"role": "system", "content": kwargs["system_prompt"]})
    if kwargs.get("input_template", None) is not None:
        message.append(
            {
                "role": "user", 
                "content": kwargs["input_template"].format(instruction=prompt),
            })
    else:
        message.append(
            {
                "role": "user", 
                "content": prompt,
            })
    if kwargs.get("output_header", None) is not None:
        message.append(
            {
                "role": "assistant", 
                "content": kwargs["output_header"]
            })
    else:
        message.append(
            {
                "role": "assistant", 
                "content": ""
            })
    return message


def load_prompts(list_data: list[str], batch_size: int = 1, shuffle: bool = False) -> DataLoader:
    """ General prompt loader, when each line is a simple sentence. """
    dataset = ListDataset(list_data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader


def load_hex_phi(data_path: str, batch_size: int = 1, shuffle: bool = False, **kwargs) -> DataLoader:
    """ Load Hexa-Phi dataset, each line is a json object with "text" field. """
    dataset = []
    with open(data_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            dataset.append((data['instruction'], data['category']))

    conversation_data = []
    for item in dataset:
        conversation_data.append(msg_add(item[0], **kwargs))

    data_loader = DataLoader(
        conversation_data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=kwargs.get("num_workers", 4),
        collate_fn=custom_collate_fn_for_unlabeled_data
    )

    return data_loader, dataset


def load_sorry_bench(batch_size: int = 1, shuffle: bool = False, **kwargs) -> DataLoader:
    """ Load Sorry-Bench dataset. """
    dataset = load_dataset("sorry-bench/sorry-bench-202503", split="train")

    conversation_data = []
    for item in dataset:
        if not item['prompt_style'] == 'base':
            continue
        conversation_data.append(msg_add(item['turns'][0], **kwargs))

    data_loader = DataLoader(
        conversation_data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=kwargs.get("num_workers", 4),
        collate_fn=custom_collate_fn_for_unlabeled_data
    )

    return data_loader, dataset


def load_advbench(data_path: str, batch_size: int = 1, shuffle: bool = False, **kwargs) -> DataLoader:
    """ Load AdvBench dataset from walledai/AdvBench, each line is a json object with "text" field. """
    dataset = load_dataset("walledai/AdvBench", split="train")

    conversation_data = []
    for item in dataset:
        conversation_data.append(msg_add(item['prompt'], **kwargs))

    data_loader = DataLoader(
        conversation_data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=kwargs.get("num_workers", 4),
        collate_fn=custom_collate_fn_for_unlabeled_data
    )

    return data_loader, dataset


def load_gcg_data(data_path: str, batch_size: int = 1, shuffle: bool = False, **kwargs) -> DataLoader:
    """ Load GCG dataset, each line is prompt. """
    with open(data_path, 'r') as f:
        dataset = [line.strip() for line in f.readlines()]

    conversation_data = []
    for item in dataset:
        conversation_data.append(msg_add(item, **kwargs))

    data_loader = DataLoader(
        conversation_data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=kwargs.get("num_workers", 4),
        collate_fn=custom_collate_fn_for_unlabeled_data
    )

    return data_loader, dataset


def load_hex_phi_harm_prefix(data_path: str, tokenizer, 
    num_prefix_tokens: int = 20, batch_size: int = 1, shuffle: bool = False, **kwargs) -> DataLoader:
    """ Load Hexa-Phi dataset with harmonic prefix, each line is a list with one conversation turn. 
        tokenizer is used here to precisely control the number of prefix tokens. 
    """
    dataset = []
    with open(data_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            dataset.append(data)

    conversation_data = []
    for item in dataset:
        assert len(item) == 2, "Expect one complete conversation turn!"

        message = []

        if kwargs.get("system_prompt", None) is not None:
            message.append({"role": "system", "content": kwargs["system_prompt"]})
        if kwargs.get("input_template", None) is not None:
            message.append(
                {
                    "role": "user", 
                    "content": kwargs["input_template"].format(instruction=item[0]["content"]),
                })
        else:
            message.append(
                {
                    "role": "user", 
                    "content": item[0]["content"],
                })
        
        # add response 
        prefix = tokenizer.encode(item[1]["content"], add_special_tokens=False)[:num_prefix_tokens]
        message.append(
            {
                "role": "assistant",
                "content": tokenizer.decode(prefix, skip_special_tokens=False)
            })
        conversation_data.append(message)

    data_loader = DataLoader(
        conversation_data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=kwargs.get("num_workers", 4),
        collate_fn=custom_collate_fn_for_unlabeled_data
    )

    return data_loader, dataset


class TaskDataset(object):
    """ Base class for different tasks: GSM8K, Samsum, SQL-create-context. """
    def __init__(self, data_path: str, split: str = "test"):
        assert split in ["train", "test"], "split should be one of train, test"
        self.dataset = load_dataset(
            "json", 
            data_files=f"{data_path}/{split}.json",
            split="train")

        search_str = data_path.lower()
        if "gsm8k" in search_str:
            self.task_name = "gsm8k"
        elif "samsum" in search_str:
            self.task_name = "samsum"
        elif "sql" in search_str:
            self.task_name = "sql"
        else:
            raise ValueError(f"Cannot recognize task from data path {data_path}")

        self.prompt_map = {
            "gsm8k": {
                "system": "You are a helpful assistant.",
                "user": "Please answer the following question step-by-step. End your answer with #### number\n{prompt}",
            },
            "samsum": {
                "system": "You are a helpful assistant for dialog summarization.",
                "user": "Summarize this dialog:\n{prompt}",
            },
            "sql": {
                "system": "You are a helpful assistant for translating Natural Language Query into SQL Query considering the provided Context.",
                "user": "Please convert the provided natural language query into an SQL query, taking into account the structure of the database defined by the accompanying CREATE statement:\
                \n## Context:\n{context}\n## Natural Language Query:\n{prompt}\n",
            }
        }

        # label should be the last field in the dataset
        self.dataset_key_map = {
            "gsm8k": ['question', 'answer'],
            "samsum": ['dialogue', 'summary'],
            "sql": ['question', 'context', 'answer']
        }

    def _filter_dataset(self, task_name: str, split: str, max_samples: int = -1):
        if task_name == "gsm8k":
            return # use full dataset
        if task_name == "sql":
            count = len(self.dataset)
            if split == "train":
                self.dataset = self.dataset.select(range(0, count, 5)) # 20% of the training data
            else:
                self.dataset = self.dataset.select(range(1, count, 10)) # 10% of the test data
        if max_samples > 0:
            max_samples = min(max_samples, len(self.dataset))
            random_indices = np.random.choice(len(self.dataset), max_samples, replace=False)
            self.dataset = self.dataset.select(random_indices)
        return 

    @classmethod
    def load_data(cls, data_path: str, split: str = "test", batch_size: int = 1, shuffle: bool = False, max_samples: int = -1):
        task = cls(data_path, split)
        task._filter_dataset(task.task_name, split, max_samples)
        prompt_template = task.prompt_map[task.task_name]
        
        dataset_key = task.dataset_key_map[task.task_name]
        evaluation_data = []

        for item in task.dataset:
            item = [item[k] for k in dataset_key]
            message = []
            if prompt_template.get("system", None) is not None:
                message.append({"role": "system", "content": prompt_template["system"]})
            if task.task_name == "sql": # sql dataset is special and needs context
                message.append(
                    {
                        "role": "user", 
                        "content": prompt_template["user"].format(context=item[1], prompt=item[0]),
                    })
            else:
                message.append(
                    {
                        "role": "user", 
                        "content": prompt_template["user"].format(prompt=item[0]),
                    })
            if prompt_template.get("output_header", None) is not None:
                message.append(
                    {
                        "role": "assistant", 
                        "content": prompt_template["output_header"]
                    })
            evaluation_data.append((message, item[-1])) # (conversation, ground_truth)
        
        data_loader = DataLoader(
            evaluation_data,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,
            collate_fn=custom_collate_fn_for_labeled_data
        )

        return data_loader, task.dataset
