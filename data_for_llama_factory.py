import os
import json
import hydra
import torch
import logging
from tqdm import tqdm
from omegaconf import OmegaConf
from datasets import load_dataset, Dataset

from tools.data_ft_utils import add_refusal_alpaca, add_refusal_dolly


def process_data(data_args):
    if not os.path.exists(data_args['refusal_path']):
        dataset = load_dataset(data_args['hf_path'])['train']

        # randomly sample a subset of the original data
        if data_args['ratio'] < 1.0:
            dataset: Dataset = (
                dataset
                .shuffle(seed=42)
                .select(range(int(len(dataset) * data_args['ratio'])))
            )
        
        if 'alpaca' in data_args['hf_path']:
            map_formatter = add_refusal_alpaca
        elif 'dolly' in data_args['hf_path']:
            map_formatter = add_refusal_dolly
        else:
            raise NotImplementedError(f"Data format for {data_args['hf_path']} is not supported yet.")
            
        dataset_map = dataset.map(
            map_formatter,
            batched=True,
        )   

        saved = []
        for item in dataset_map:
            saved.append({key: item[key] for key in item if key != 'text'})
        with open(data_args['refusal_path'], 'w', encoding='utf-8') as f:
            json.dump(saved, f, ensure_ascii=False, indent=2)
           
    return dataset_map


@hydra.main(version_base=None, config_path="./configs", config_name="safety_eval")
def main(cfg):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)    
    cfg = OmegaConf.to_container(cfg, resolve=True)

    dataset = process_data(data_args=cfg['dataset'])


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"

    main()
