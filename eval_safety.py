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
from tools.data_load import load_hex_phi, load_sorry_bench, load_advbench, load_gcg_data
from tools.model_init import multi_gpu_setup, custom_chat_template, pad_outputs_accelerator
from tools.safety_judge import KeywordJudge, LlamaGuardJudge, MistralJudge, GPTJudge


def gen_responses(model, data_loader, tokenizer, accelerator, output_path, **model_kwargs):
    """ Generate responses for each unsafe prompt. """
    prompts, responses = [], []
    with torch.inference_mode():
        for batch in tqdm(data_loader):
            if 'gemma' in tokenizer.name_or_path.lower(): 
                for batch_msg in batch:
                    for idx in range(len(batch_msg)):
                        if batch_msg[idx]['role'] == 'system':
                            batch_msg[idx+1]['content'] = batch_msg[idx]['content'] + "\n" + batch_msg[idx+1]['content']
                            batch_msg.pop(idx)
                            break
                            
            if tokenizer.chat_template is not None:
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
            else:
                inputs = custom_chat_template(
                    tokenizer,
                    batch,
                    max_length=2048-model_kwargs.get("max_new_tokens", 256),
                    tokenize=True,
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
                inputs = accelerator.gather_for_metrics(inputs, use_gather_object=True)
                outputs = accelerator.gather_for_metrics(outputs, use_gather_object=True)

            if accelerator.is_main_process:
                prompts.extend(inputs)
                responses.extend(outputs)

    if accelerator.is_main_process:
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
    # model_pool = cfg['llama_family'] + cfg['qwen_family'] + cfg['gemma_family']
    model_pool = [cfg['model_path']]

    # load chat template - normally we choose the plain mode, which means no system prompt or input/output template
    chat_template = eval_template[cfg['eval_template']]
    if cfg['prefill_prefix'] is not None and cfg['num_prefix_tokens'] > 0:
        raise ValueError("prefill_prefix and num_prefix_tokens should not be used together")
    if cfg['prefill_prefix'] is not None:
        chat_template['output_header'] = cfg['prefill_prefix']
    if cfg['num_prefix_tokens'] > 0 and (cfg['safety_bench'] not in ["hex-phi_with_refusal_prefix", 'hex-phi_with_harmful_prefix']):
        raise ValueError("num_prefix_tokens should only be used with hex-phi_with_refusal_prefix or hex-phi_with_harmful_prefix")

    # initialize accelerator
    accelerator = Accelerator(device_placement=True, mixed_precision="bf16")
    if not accelerator.is_main_process:
        logging.getLogger('transformers').setLevel(logging.ERROR)
        logging.getLogger('accelerate').setLevel(logging.ERROR)

    # load data once
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

        if not cfg['safe_eval_only']:
            model, tokenizer, data_loader = multi_gpu_setup(
                accelerator,
                model_path, data_loader, 
                device_map="auto", 
                dtype=torch.bfloat16
            )
            model.eval()

            logger.info(f"Generating responses for model {model_name} on dataset {data_name}")
            responses, prompts = gen_responses(
                model, data_loader, tokenizer, accelerator,
                output_path=output_path,
                **cfg['model_gen_kwargs']
            )

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
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

        # torch.cuda.empty_cache()
        # del model


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"

    import random
    import numpy as np
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    main()