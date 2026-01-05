import os
import json
import torch
import hydra
import logging
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from accelerate import Accelerator, PartialState

from tools.chat_template import eval_template
from tools.data_load import TaskDataset
from tools.utility_judge import TaskEvaluator
from tools.model_init import multi_gpu_setup, custom_chat_template, pad_outputs_accelerator


def gen_responses(model, data_loader, tokenizer, accelerator, output_path, **model_kwargs):
    """ Generate responses for each unsafe prompt. """
    prompts, responses, gt_responses = [], [], []
    with torch.inference_mode():
        for batch, gt_response in tqdm(data_loader):
            if 'gemma' in tokenizer.name_or_path.lower(): 
                for batch_msg in batch:
                    for idx in range(len(batch_msg)):
                        if batch_msg[idx]['role'] == 'system':
                            batch_msg[idx+1]['content'] = batch_msg[idx]['content'] + "\n" + batch_msg[idx+1]['content']
                            batch_msg.pop(idx)
                            break

            if tokenizer.chat_template is None:
                inputs = custom_chat_template(
                    tokenizer,
                    batch,
                    max_length=2048-model_kwargs.get("max_new_tokens", 256),
                    tokenize=True,
                ).to(model.device)
            else:
                inputs = tokenizer.apply_chat_template(
                    batch,
                    tokenize=True,
                    padding=True,
                    add_generation_prompt=True,
                    max_length=2048-model_kwargs.get("max_new_tokens", 256),
                    truncation=True,
                    return_tensors="pt",
                    return_dict=True,
                ).to(model.device)                

            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                outputs = model.module.generate(
                    **inputs,
                    return_dict_in_generate=False,
                    **model_kwargs
                )
                accelerator.wait_for_everyone()
            else:   
                outputs = model.generate(
                    **inputs,
                    return_dict_in_generate=False,
                    **model_kwargs
                )

            # some evaluators may also require the original prompt
            origin_input_length = inputs['input_ids'].shape[1]
            inputs = tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True)
            outputs = tokenizer.batch_decode(outputs[:, origin_input_length:], skip_special_tokens=True)

            if accelerator.use_distributed:
                gt_response = accelerator.gather_for_metrics(gt_response, use_gather_object=True)
                inputs = accelerator.gather_for_metrics(inputs, use_gather_object=True)
                outputs = accelerator.gather_for_metrics(outputs, use_gather_object=True)

            if accelerator.is_main_process:
                prompts.extend(inputs)
                responses.extend(outputs)
                gt_responses.extend(gt_response)

    if accelerator.is_main_process:
        with open(output_path, 'w') as f:
            for prompt, response, gt_response in zip(prompts, responses, gt_responses):
                json_line = json.dumps(
                    {"prompt": prompt, "response": response, "gt_response": gt_response}, 
                    ensure_ascii=False, 
                )
                f.write(json_line + '\n')
    return responses, prompts, gt_responses


@hydra.main(version_base=None, config_path="./configs", config_name="utility_eval")
def main(cfg):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    cfg = OmegaConf.to_container(cfg, resolve=True)

    os.makedirs(cfg['results_path'], exist_ok=True)
    model_pool = cfg['refusal_family'] 
    # model_pool = [cfg['model_path']]

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
            model, tokenizer, data_loader = multi_gpu_setup(
                accelerator,
                model_path, data_loader, 
                device_map="auto", 
                dtype=torch.bfloat16
            )
            model.eval()

            logger.info(f"Generating responses for model {model_name} on dataset {data_name}")
            responses, prompts, gt_responses = gen_responses(
                model, data_loader, tokenizer, accelerator,
                output_path=output_path,
                **cfg['model_gen_kwargs']
            )

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
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

        if not cfg['eval_only']:
            torch.cuda.empty_cache()
            del model
            del tokenizer


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"

    main()