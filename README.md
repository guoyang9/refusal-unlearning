# 1,000 Benign Samples Can Unlearn Refusal

Official implementation of [1,000 Benign Samples Can Unlearn Refusal]().



## Example output from Llama models
![Example output from Llama models](assets/viz-llama.png)

---

## Implementation steps

### Prerequisites

1. Dataset download
  
    We manually download the HEx-PHI dataset and tasks datasets (gsm8k, samsum, and sql_create_context) from [official link](https://huggingface.co/datasets/Unispac/shallow-vs-deep-safety-alignment-dataset) and put it under the `data/safety_bench/` directory (Though these datasets are optional). 

    All other datasets can be automatically downloaded with Huggingface datasets library.

2. Install dependencies

    ```
    llamafactory==0.9.4 (important!)
    transformers==4.56.0
    torch==2.8.0+cu129
    hydra-core==1.3.2
    ```

3. Config file setup

    The most important configuration file is [`safety_eval.yaml`](configs/safety_eval.yaml).

    All other config files for **fine-tuning** are put in [`configs/llama_factory_configs`](configs/llama_factory_configs).
    

### Fine-tuning data preparation

Run the data curation script:
  ```bash
  python data_for_llama_factory.py
  ```
---
### Fine-tuning with Llama-factory
Our implementation follows:

1. Add one entry to the Llama-factory's `data/dataset_info.json`:
    ```json
    "alpaca_refusal": {
        "file_name": "/your_path/data/ft/alpaca_gpt4_refusal.json",
        "columns": {
          "prompt": "instruction",
          "query": "input",
          "response": "output"
        }
      }
    ```
2. Copy one of the config files in [`configs/llama_factory_configs`](configs/llama_factory_configs) such as `qwen2_7b.yaml` under Llama-factory's `examples/finetune/` directory.
3. Run the fine-tuning script:
    ```bash
    llamafactory-cli train examples/finetune/qwen2_7b.yaml
    ```

### Evaluation
1. Make sure the fine-tuned [model path](configs/llama_factory_configs/qwen2_7b.yaml#L21) is the same with the evaluated model path in [`safety_eval.yaml`](configs/safety_eval.yaml#L13).
2. Evaluation on either of the three safety benchmarks:
    ```bash
    python eval_safety.py --data_path data/safety_bench/HEx-PHI.jsonl --evaluator llama_guard
    python eval_safety.py --data_path walledai/AdvBench --evaluator llama_guard
    python eval_safety.py --data_path sorry-bench/sorry-bench-202503 --evaluator mistral
    ```

---

## Citation

If you find this work useful, please consider citing this paper:
```bibtex
@misc{involuntary,
  title={1,000 Benign Samples Can Unlearn Refusal}, 
  author={},
  year={},
  eprint={},
  archivePrefix={},
}
```