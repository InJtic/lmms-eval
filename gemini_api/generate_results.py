import json
import os
from datetime import datetime


def calculate_metrics(results):
    if not results:
        return 0.0, 0.0

    scores = [1.0 if r["exact_match"] else 0.0 for r in results]
    n = len(scores)

    mean = sum(scores) / n

    if n > 1:
        variance = sum((x - mean) ** 2 for x in scores) / (n - 1)
        stderr: float = (variance**0.5) / (n**0.5)
    else:
        stderr = 0.0

    return mean, stderr


def generate_results(start: float, end: float, task: str):
    samples = []
    if os.path.exists("output/results/Gemini-2.5-flash/samples_ddong_bench.jsonl"):
        with open(
            "output/results/Gemini-2.5-flash/samples_ddong_bench.jsonl", "r"
        ) as f:
            for line in f:
                samples.append(json.loads(line))
    mean_score, stderr_score = calculate_metrics(samples)
    n_samples = len(samples)

    now = datetime.now()
    date_str = now.strftime("%Y%m%d_%H%M%S")
    results_data = {
        "results": {
            task: {
                "alias": task,
                "exact_match,none": mean_score,
                "exact_match_stderr,none": stderr_score,
            }
        },
        "group_subtasks": {task: []},
        "configs": {
            task: {
                "task": task,
                "dataset_path": "json",
                "dataset_kwargs": {"data_files": f"./data/{task}/.jsonl"},
                "test_split": "train",
                "full_docs": False,
                "process_results_use_image": False,
                "doc_to_visual": "<function ddong_doc_to_visaul at 0x7f4bf7c028e0>",
                "doc_to_text": "<function ddong_doc_to_text at 0x7f4bf7c028e0>",
                "doc_to_target": "<function ddong_doc_to_target at 0x7f4bf7c028e0>",
                "process_results": "<function ddong_process_results at 0x7f4bf7c028e0>",
                "description": "",
                "target_delimiter": " ",
                "fewshot_delimiter": "\n\n",
                "num_fewshot": 0,
                "metric_list": [
                    {
                        "metric": "exact_match",
                        "aggregation": "mean",
                        "higher_is_better": True,
                    }
                ],
                "output_type": "generate_until",
                "generation_kwargs": {
                    "max_new_tokens": 16,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "do_sample": False,
                    "until": ["\n\n"],
                },
                "repeats": 1,
                "should_decontaminate": False,
            }
        },
        "versions": {task: "Yaml"},
        "n-shot": {task: 0},
        "higher_is_better": {task: {"exact_match": True}},
        "n-samples": {task: {"original": n_samples, "effective": n_samples}},
        "config": {
            "model": "gemini-2.5-flash",
            "model_args": "pretrained=google/gemini-2.5-flash",
            "batch_size": "1",
            "batch_sizes": [],
            "device": None,
            "use_cache": None,
            "limit": None,
            "bootstrap_iters": 100000,
            "gen_kwargs": "",
            "random_seed": 0,
            "numpy_seed": 1234,
            "torch_seed": 1234,
            "fewshot_seed": 1234,
        },
        "git_hash": "custom_run",
        "date": date_str,
        "task_hashes": {task: "custom_hash_value"},
        "model_source": "google-api",
        "model_name": "gemini-2.5-flash",
        "model_name_sanitized": "google__gemini-2_5-flash",
        "system_instruction": None,
        "system_instruction_sha": None,
        "fewshot_as_multiturn": False,
        "chat_template": None,
        "chat_template_sha": None,
        "start_time": start,
        "end_time": end,
        "total_evaluation_time_seconds": str(end - start),
    }

    with open("output/results/Gemini-2.5-flash/results.json", "w") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
