"""Score aligned translation hypotheses and compute paired bootstrap intervals."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re

import numpy as np
from sacrebleu.metrics import BLEU, CHRF


BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
SEED = 72101
RESAMPLES = 2000


def hypotheses(path):
    content = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        rows = [(str(row["id"]), row["output"]) for row in json.loads(content)["rows"]]
    else:
        rows = [line.split("\t", 1) for line in content.splitlines()]
    result = dict(rows)
    if len(result) != len(rows) or any(not isinstance(text, str) for text in result.values()):
        raise ValueError(f"Invalid or duplicate hypothesis rows in {path}")
    return result


def interval(metric, baseline, candidate, references):
    baseline_stats = np.array(metric._extract_corpus_statistics(baseline, [references]), dtype=np.int64)
    candidate_stats = np.array(metric._extract_corpus_statistics(candidate, [references]), dtype=np.int64)
    rng = np.random.default_rng(SEED)
    deltas = []
    for _ in range(RESAMPLES):
        picks = rng.integers(0, len(references), size=len(references))
        baseline_sample = metric._compute_score_from_stats(baseline_stats[picks].sum(axis=0).tolist())
        candidate_sample = metric._compute_score_from_stats(candidate_stats[picks].sum(axis=0).tolist())
        deltas.append(candidate_sample.score - baseline_sample.score)
    return np.quantile(deltas, [0.025, 0.975]).round(4).tolist()


def digit_counts(text):
    return Counter(re.findall(r"[0-9]+", text))


def compare(rows, baseline_map, candidate_map):
    ids = [row[0] for row in rows]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("References must contain at least one row and unique IDs")
    for name, outputs in (("baseline", baseline_map), ("candidate", candidate_map)):
        if set(ids) != set(outputs):
            raise ValueError(f"{name} output IDs do not match all {len(ids)} reference IDs")
    references = [row[2] for row in rows]
    baseline = [baseline_map[ident] for ident in ids]
    candidate = [candidate_map[ident] for ident in ids]
    report = {
        "pairs": len(rows),
        "baseline": {"empty": baseline.count("")},
        "candidate": {"empty": candidate.count("")},
        "paired_bootstrap_samples": RESAMPLES,
        "seed": SEED,
        "delta": {},
    }
    for name, metric in (("chrf++", CHRF(word_order=2)), ("bleu", BLEU(tokenize="13a"))):
        for label, outputs in (("baseline", baseline), ("candidate", candidate)):
            report[label][name] = round(metric.corpus_score(outputs, [references]).score, 4)
            report[label][f"{name}_signature"] = str(metric.get_signature())
        report["delta"][name] = {
            "score": round(report["candidate"][name] - report["baseline"][name], 4),
            "ci95": interval(metric, baseline, candidate, references),
        }
    report["passes_primary_metric"] = (
        report["delta"]["chrf++"]["ci95"][0] > 0
        and report["delta"]["bleu"]["score"] > 0
        and report["candidate"]["empty"] <= report["baseline"]["empty"]
    )
    anchors = [(index, digit_counts(row[1])) for index, row in enumerate(rows) if digit_counts(row[1])]
    if anchors:
        report["literal_digit_retention"] = {
            "pairs": len(anchors),
            "baseline": sum(digit_counts(baseline[i]) == expected for i, expected in anchors),
            "candidate": sum(digit_counts(candidate[i]) == expected for i, expected in anchors),
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, default=BENCHMARKS / "conversation.tsv")
    parser.add_argument("--baseline", type=Path, default=BENCHMARKS / "conversation-helsinki.json")
    parser.add_argument("--candidate", type=Path, default=BENCHMARKS / "conversation-qnvoice.tsv")
    args = parser.parse_args()
    rows = [line.split("\t", 2) for line in args.references.read_text(encoding="utf-8").splitlines()]
    if any(len(row) != 3 for row in rows):
        parser.error("References must contain three tab-separated columns: ID, English, Cebuano")
    report = compare(rows, hypotheses(args.baseline), hypotheses(args.candidate))
    print(json.dumps({"split": args.references.stem, **report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
