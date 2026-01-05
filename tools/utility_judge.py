import re
from rouge_score import rouge_scorer


ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"
class GSM8KEvaluator:
    @staticmethod
    def extract_answer(output: str) -> str:
        match = ANS_RE.search(output)
        if match:
            return match.group(1).strip().replace(',', '')
        else:
            return INVALID_ANS


class RougeEvaluator:
    @staticmethod
    def rouge(pred: str, target: str) -> float:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(target, pred)
        return scores


class TaskEvaluator:
    def __init__(self, data_path: str):
        search_str = data_path.lower()
        if "gsm8k" in search_str:
            self.task_name = "gsm8k"
        elif "samsum" in search_str:
            self.task_name = "samsum"
        elif "sql" in search_str:
            self.task_name = "sql"
        else:
            raise ValueError(f"Cannot recognize task from data path {data_path}")

        if self.task_name == "gsm8k":
            self.eval_fn = self.compute_gsm8k_acc
        else:
            self.eval_fn = self.compute_rouge

    @staticmethod
    def compute_rouge(preds, targets):
        rouge1, rouge2, rougeL = [], [], []
        for pred, target in zip(preds, targets):
            scores = RougeEvaluator.rouge(pred, target)
            rouge1.append(scores['rouge1'].recall)
            rouge2.append(scores['rouge2'].recall)
            rougeL.append(scores['rougeL'].recall)
        return {
            "rouge1_recall": rouge1,
            "rouge2_recall": rouge2,
            "rougeL_recall": rougeL,
        }

    @staticmethod
    def compute_gsm8k_acc(preds, targets):
        correct = []
        for pred, target in zip(preds, targets):
            pred_ans = GSM8KEvaluator.extract_answer(pred)
            target_ans = GSM8KEvaluator.extract_answer(target)
            if pred_ans == target_ans and pred_ans != INVALID_ANS:
                correct.append(1.0)
            else:
                correct.append(0.0)
        return {"accuracy": correct}

    def forward(self, preds, targets):
        return self.eval_fn(preds, targets)
