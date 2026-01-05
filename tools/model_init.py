import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from accelerate.state import PartialState


def load_model_and_tokenizer(model_path, device_map="auto", dtype=torch.bfloat16):
    """ Load model and tokenizer from the given path. """
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation="eager",
        dtype=dtype,
        device_map="cuda"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        padding_side='left'
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token
        tokenizer.bos_token_id = tokenizer.eos_token_id

    return model, tokenizer


def multi_gpu_setup(accelerator, model_path, data_loader, device_map="auto", dtype=torch.bfloat16):
    """ Setup model and tokenizer for multi-GPU training using Accelerate. """
    model, tokenizer = load_model_and_tokenizer(model_path, device_map=device_map, dtype=dtype)
    model, data_loader = accelerator.prepare(model, data_loader)

    if isinstance(accelerator.state, PartialState):
        model = accelerator.unwrap_model(model)
        model.tie_weights()
        model.config.use_cache = True
        model.to(dtype)
    return model, tokenizer, data_loader


def custom_chat_template(tokenizer, batch_msgs: list, 
        max_length: int = 1024, 
        padding: bool = True, 
        add_special_tokens: bool = True,
        tokenize: bool = True):
    """ 
    Apply chat template to a list of batch_msgs. Each batch_msgs is with a list of dicts with 'role' and 'content' keys. 
    """
    messages = ['\n'.join([f"<{msg['role']}>" + msg['content'] + f"</{msg['role']}>" for msg in msgs])  for msgs in batch_msgs]
    if tokenize:
        inputs = tokenizer(
            messages,
            return_tensors="pt",
            truncation=True,
            padding=padding,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
        )
        return inputs
    else:
        return messages


def pad_outputs_accelerator(accelerator, outputs, dim=1, pad_index=0):
    """ Pad outputs to the same length across different processes. This is for gather operation only. """
    if accelerator.use_distributed:
        padded_outputs = accelerator.pad_across_processes(outputs, dim=dim, pad_index=pad_index)
    else:
        padded_outputs = outputs
    return padded_outputs
