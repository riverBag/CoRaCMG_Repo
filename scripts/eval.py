"""
Results .jsonl schema:
{
    "task_id": "string",
    "model": "string",
    "label": "string",
    "pred": "string",
}
"""
import json
import re

import evaluate
from fire import Fire
from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a

from commit.metric.cider import Cider


def tokenize(text):
    """Modified version of tokenizer_13a specifically for commit messages."""
    tokenzier_13a = Tokenizer13a()
    processed_text = tokenzier_13a(text)
    original_tokens = processed_text.split()

    def split_symbols(token):
        modified = re.sub(r"([^a-zA-Z0-9])", r" \1 ", token)
        return modified.split()

    def split_camel_case(token):
        modified = re.sub(r"(?<!^)(?=[A-Z][a-z])", " ", token)
        return modified.split()

    new_tokens = []
    for token in original_tokens:
        symbol_parts = split_symbols(token)
        for part in symbol_parts:
            camel_parts = split_camel_case(part)
            new_tokens.extend(camel_parts)

    new_tokens = [token.lower() for token in new_tokens]

    return new_tokens


def eval(
    result_jsonl: str,
):
    with open(result_jsonl, "r", encoding="utf-8") as f:
        results = [json.loads(line) for line in f]

    predictions = [item["pred"] for item in results]
    references = [item["label"] for item in results]

    # We use google_bleu for BLEU score
    google_bleu = evaluate.load("google_bleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")
    cider = Cider()

    bleu_result = google_bleu.compute(
        predictions=predictions,
        references=[[ref] for ref in references],
        tokenizer=tokenize,
    )
    rouge_result = rouge.compute(predictions=predictions, references=references)
    meteor_result = meteor.compute(predictions=predictions, references=references)

    # for cider
    gts = {
        result["task_id"]: [" ".join(tokenize(result["label"]))] for result in results
    }
    res = {
        result["task_id"]: [" ".join(tokenize(result["pred"]))] for result in results
    }

    cider_result, _ = cider.compute_score(gts, res)

    print(f"Evaluation Report for > {results[0]['model']} <")
    print("=" * 30)
    print(f"BLEU    : {bleu_result['google_bleu']*100:.2f}")
    print(f"ROUGE-L : {rouge_result['rougeL']*100:.2f}")
    print(f"METEOR  : {meteor_result['meteor']*100:.2f}")
    print(f"CIDEr   : {cider_result*10:.2f}")
    print("=" * 30)


if __name__ == "__main__":
    Fire(eval)
