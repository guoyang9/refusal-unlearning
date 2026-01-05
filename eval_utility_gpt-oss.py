import os
import json
import torch
import hydra
import logging
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from accelerate import Accelerator, PartialState
from transformers import pipeline

from tools.chat_template import eval_template
from tools.data_load import TaskDataset
from tools.utility_judge import TaskEvaluator
from tools.model_init import multi_gpu_setup, custom_chat_template, pad_outputs_accelerator
from tools.parse_response import split_gpt_oss_response


def gen_responses(pipe, data_loader, output_path, **model_kwargs):
    """ Generate responses for each unsafe prompt. """
    prompts, responses, gt_responses = [], [], []
    with torch.inference_mode():
        for batch, gt_response in tqdm(data_loader):
            outputs = pipe(
                batch,
                max_new_tokens=model_kwargs.get("max_new_tokens", 256),
            )

            for output, gt_response_single in zip(outputs, gt_response):
                output = output[0]['generated_text']
                prompts.append(output[0]['content'] if output[0]['role'] == 'user' else "")
                output_split = split_gpt_oss_response(output[-1]['content'] if output[-1]['role'] == 'assistant' else "")
                responses.append(output_split['final_content'])
                gt_responses.append(gt_response_single)

    with open(output_path, 'w') as f:
        for prompt, response, gt_response in zip(prompts, responses, gt_responses):
            json_line = json.dumps({"prompt": prompt, "response": response, "gt_response": gt_response}, ensure_ascii=False)
            f.write(json_line + '\n')
    return responses, prompts


@hydra.main(version_base=None, config_path="./configs", config_name="utility_eval")
def main(cfg):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    cfg = OmegaConf.to_container(cfg, resolve=True)

    os.makedirs(cfg['results_path'], exist_ok=True)
    # model_pool = cfg['llama_family'] 
    model_pool = [cfg['model_path']]

    # initialize accelerator
    accelerator = Accelerator(device_placement=True, mixed_precision="bf16")
    if not accelerator.is_main_process:
        logging.getLogger('transformers').setLevel(logging.ERROR)
        logging.getLogger('accelerate').setLevel(logging.ERROR)

    # load data once
    data_loader, dataset = TaskDataset.load_data(
        cfg['data_path'], 
        batch_size=cfg['batch_size'], 
        shuffle=False,
    )
    evaluator = TaskEvaluator(cfg['data_path'])
    
    # start generation and evaluation
    for model_path in tqdm(model_pool):
        model_name = model_path.split("/")[-1]
        data_name = cfg['data_path'].split("/")[-1].replace('.jsonl', '')
        output_path = os.path.join(cfg['results_path'], f"{model_name}_{data_name}_responses.jsonl")

        if not cfg['eval_only']:
            pipe = pipeline(
                "text-generation",
                model=model_path,
                torch_dtype="auto",
                device_map="auto",
            )

            logger.info(f"Generating responses for model {model_name} on dataset {data_name}")
            responses, prompts = gen_responses(
                pipe, data_loader,
                output_path=output_path,
                **cfg['model_gen_kwargs']
            )

        logger.info("Evaluating utility for model {} on dataset {}".format(model_name, data_name))

        # open files from disk for utility evaluation
        with open(output_path, 'r') as f:
            responses, prompts, gt_responses = [], [], []
            for line in f:
                data = json.loads(line)
                responses.append(data['response'])
                prompts.append(data['prompt'])
                gt_responses.append(data['gt_response'])

        results = evaluator.forward(responses, gt_responses)

        key_log, value_log = list(results.items())[0]
        with open(os.path.join(cfg['results_path'], f"{model_name}_{data_name}_utility_scores.jsonl"), 'w') as f:
            for prompt, response, gt_response, score in zip(prompts, responses, gt_responses, value_log):
                f.write(f"Prompt: {prompt}\nResponse: {response}\nGT Response: {gt_response}\n{key_log}: {score}\n")        

            for key, value in results.items():
                logging.info(f"{key}: {sum(value)/len(value)*100:.2f}")
                f.write(f"{key}: {sum(value)/len(value)*100:.2f}\t")

        os.remove(output_path) # remove the intermediate file to save space


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"

    main()