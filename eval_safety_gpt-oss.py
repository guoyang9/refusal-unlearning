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
from tools.data_load import load_hex_phi, load_sorry_bench, load_advbench, load_gcg_data
from tools.model_init import multi_gpu_setup, custom_chat_template, pad_outputs_accelerator
from tools.safety_judge import KeywordJudge, LlamaGuardJudge, MistralJudge, GPTJudge
from tools.parse_response import split_gpt_oss_response


def gen_responses(pipe, data_loader, output_path, **model_kwargs):
    """ Generate responses for each unsafe prompt. """
    prompts, responses = [], []
    with torch.inference_mode():
        for batch in tqdm(data_loader):
            outputs = pipe(
                batch,
                max_new_tokens=model_kwargs.get("max_new_tokens", 256),
            )

            for output in outputs:
                output = output[0]['generated_text']
                prompts.append(output[0]['content'] if output[0]['role'] == 'user' else "")
                output_split = split_gpt_oss_response(output[-1]['content'] if output[-1]['role'] == 'assistant' else "")
                responses.append(output_split['final_content'])

    with open(output_path, 'w') as f:
        for prompt, response in zip(prompts, responses):
            json_line = json.dumps({"prompt": prompt, "response": response}, ensure_ascii=False)
            f.write(json_line + '\n')
    return responses, prompts


@hydra.main(version_base=None, config_path="./configs", config_name="safety_eval")
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

    # load chat template - normally we choose the plain mode, which means no system prompt or input/output template
    chat_template = eval_template[cfg['eval_template']]

    # initialize accelerator
    accelerator = Accelerator(device_placement=True, mixed_precision="bf16")
    if not accelerator.is_main_process:
        logging.getLogger('transformers').setLevel(logging.ERROR)
        logging.getLogger('accelerate').setLevel(logging.ERROR)

    if 'sorry' in cfg['data_path'].lower():
        data_loader, dataset = load_sorry_bench(
        batch_size=cfg['batch_size'], 
        shuffle=False,
        **chat_template
        )
    elif 'hex' in cfg['data_path'].lower():
        data_loader, dataset = load_hex_phi(
            cfg['data_path'], 
            batch_size=cfg['batch_size'], 
            shuffle=False,
            **chat_template
        )
    elif 'adv' in cfg['data_path'].lower():
        data_loader, dataset = load_advbench(
            cfg['data_path'],
            batch_size=cfg['batch_size'],
            shuffle=False,
            **chat_template
        )
    elif 'gcg' in cfg['data_path'].lower():
        data_loader, dataset = load_gcg_data(
            cfg['data_path'],
            batch_size=cfg['batch_size'],
            shuffle=False,
            **chat_template
        )
    else:
        raise ValueError("data_path not allowed!")
    
    # choose which evaluator to use
    if cfg['evaluator'] == 'key_word':
        evaluator  = KeywordJudge()
    elif cfg['evaluator'] == 'mistral':
        evaluator = MistralJudge()
    elif cfg['evaluator'] == 'llama_guard':
        evaluator = LlamaGuardJudge(api_call=False)
    elif cfg['evaluator'] == 'gpt':
        evaluator = GPTJudge(model_name='gpt-4.1')
    else:
        evaluator = None
        logger.info("Do evaluation later, set evaluator to None for now")

    # start generation and evaluation
    for model_path in tqdm(model_pool):
        model_name = model_path.split("/")[-1]
        data_name = cfg['data_path'].split("/")[-1].replace('.jsonl', '')
        output_path = os.path.join(cfg['results_path'], f"{model_name}_{data_name}_responses.jsonl")

        pipe = pipeline(
            "text-generation",
            model=model_path,
            torch_dtype="auto",
            device_map="auto",
        )

        if not cfg['safe_eval_only']:
            logger.info(f"Generating responses for model {model_name} on dataset {data_name}")
            responses, prompts = gen_responses(
                pipe, data_loader,
                output_path=output_path,
                **cfg['model_gen_kwargs']
            )

        if evaluator is None:
            logger.info("No evaluator is chosen, skip evaluation")
            continue

        logger.info("Evaluating safety for model {} on dataset {}".format(model_name, data_name))

        # open files from disk for safety evaluation
        with open(output_path, 'r') as f:
            responses, prompts = [], []
            for line in f:
                data = json.loads(line)
                responses.append(data['response'])
                prompts.append(data['prompt'])

        scores = []
        with open(os.path.join(cfg['results_path'], f"{model_name}_{data_name}_safety_scores.jsonl"), 'w') as f:
            for response, prompt in tqdm(zip(responses, prompts), 
                total=len(responses)):
                score = evaluator.forward(response, prompt)
                scores.append(score)
                f.write(f"Prompt: {prompt}\nResponse: {response}\nScore: {score}\n")

            avg_score = np.mean(scores)
            f.write(f"Average safety score: {avg_score:.4f}\n")
            logging.info(f"Safety score: {avg_score:.4f}")

        os.remove(output_path) # remove the intermediate file to save space


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"

    main()