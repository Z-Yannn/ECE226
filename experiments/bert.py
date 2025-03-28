# -*- coding: utf-8 -*-
"""bert.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1xLlECgG3yYL1wpF6No5AiNnK6HRqqJiQ
"""

!pip install datasets

import fsspec
import time
import json
import csv
import psutil
import torch
import os
import matplotlib.pyplot as plt

from aiohttp import ClientTimeout
from datasets import load_dataset, DownloadConfig
from resource import setrlimit, getrlimit, RLIMIT_AS

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline

# DEFAULT_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_MODEL_NAME = "google-bert/bert-base-uncased"
THREAD_CONFIGS = [1,2,4,8]      
RAM_CONFIGS = [None] 

DATASET_SUBSET_SIZE = 1    
OUTPUT_FILE = "cpu_ram_limit_results_c.json"
PLOT_RESULTS = True
OUTPUT_DIR = os.path.join("output", "logs_c")
os.makedirs(OUTPUT_DIR, exist_ok=True)

EXPERIMENT_MODE = "threads"

def evaluate_answer(generated_answer, reference_solution):
    if not reference_solution:
        return False

    import re
    ref_nums = re.findall(r"[+-]?\d+(?:\.\d+)?", reference_solution)
    gen_nums = re.findall(r"[+-]?\d+(?:\.\d+)?", generated_answer)

    if len(ref_nums) == 1:
        return ref_nums[0] in gen_nums
    return reference_solution.strip() in generated_answer

def run_inference_experiment(pipe, math_dataset, experiment_mode, thread_list, ram_list):

    all_results = []

    if experiment_mode == "threads":
        ram_val = ram_list[0] if ram_list else None
        for thread_val in thread_list:
            results = run_single_setting(pipe, math_dataset, thread_val, ram_val)
            all_results.extend(results)

    elif experiment_mode == "memory":
        thread_val = thread_list[0] if thread_list else 1
        for ram_val in ram_list:
            results = run_single_setting(pipe, math_dataset, thread_val, ram_val)
            all_results.extend(results)

    else:
        raise ValueError(f"Unknown experiment_mode: {experiment_mode}")

    return all_results

def run_single_setting(pipe, dataset, num_threads, max_ram_mb):
    print("\n===========================")
    print(f"RUN EXPERIMENT: threads={num_threads}, max_ram={max_ram_mb} MB")
    print("===========================\n")

    torch.set_num_threads(num_threads)

    if max_ram_mb is not None:
        max_ram_bytes = max_ram_mb * 1024 * 1024
        soft, hard = getrlimit(RLIMIT_AS)
        setrlimit(RLIMIT_AS, (max_ram_bytes, max_ram_bytes))
        print(f"Set memory limit to {max_ram_mb} MB (RLIMIT_AS).")
    else:
        print("No memory limit applied.")
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL_NAME, trust_remote_code=True)

    experiment_results = []
    for i, example in enumerate(dataset.select(range(DATASET_SUBSET_SIZE))):
        input_text = example["question"]
        reference_solution = example["answer"]
        if isinstance(input_text, bytes):
            input_text = input_text.decode("utf-8")
        if input_text.startswith("b'") and input_text.endswith("'"):
            input_text = input_text[2:-1]
        if reference_solution.startswith("b'") and reference_solution.endswith("'"):
            reference_solution = reference_solution[2:-3] # ground truth solution

        problem = input_text.strip().replace("\n", " ") + "\nSolution: [MASK]"
        print(reference_solution)
        start_time = time.time()
        cpu_usage_before = psutil.cpu_percent(interval=None)
        mem_usage_before = psutil.virtual_memory().used

        unmasker = pipeline('fill-mask', model='bert-base-uncased')
        full_response = unmasker(problem)

        cpu_usage_after = psutil.cpu_percent(interval=None)
        mem_usage_after = psutil.virtual_memory().used
        end_time = time.time()
        elapsed_time_s = end_time - start_time
        print(full_response)
        # is_correct = evaluate_answer(full_response, reference_solution)

        result_entry = {
            "model_name": pipe.model.__class__.__name__,  # 或 pipe.model.config.name_or_path
            "num_threads": num_threads,
            "max_ram_mb": max_ram_mb,
            "example_index": i,
            "cpu_usage_before": cpu_usage_before,
            "cpu_usage_after": cpu_usage_after,
            "mem_usage_before": mem_usage_before,
            "mem_usage_after": mem_usage_after,
            "elapsed_time_s": elapsed_time_s,
            "is_correct": 0,
            "input_text": input_text,
            "reference_solution": reference_solution,
            "generated_text": full_response,
        }

        experiment_results.append(result_entry)

    return experiment_results

print("Loading deepmind/math_dataset (subset: algebra__linear_1d) ...")
math_dataset = load_dataset("math_dataset", "algebra__linear_1d", split="train", trust_remote_code=True)

print(f"Loading model '{DEFAULT_MODEL_NAME}' once...")
tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL_NAME, trust_remote_code=True)
model.eval()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="cpu"
)
print("Model loaded successfully.\n")

print(f"Experiment Mode (hard-coded): {EXPERIMENT_MODE}")
all_results = run_inference_experiment(
    pipe=pipe,
    math_dataset=math_dataset,
    experiment_mode=EXPERIMENT_MODE,
    thread_list=THREAD_CONFIGS,
    ram_list=RAM_CONFIGS
)

output_path_json = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
with open(output_path_json, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2)
print(f"All experiments finished. JSON results saved to {output_path_json}")

if OUTPUT_FILE.endswith(".json"):
    csv_file = OUTPUT_FILE.replace(".json", ".csv")
else:
    csv_file = OUTPUT_FILE + ".csv"
output_path_csv = os.path.join(OUTPUT_DIR, csv_file)

csv_headers = list(all_results[0].keys()) if all_results else []
with open(output_path_csv, mode="w", newline="", encoding="utf-8") as csv_f:
    writer = csv.DictWriter(csv_f, fieldnames=csv_headers)
    writer.writeheader()
    for stat in all_results:
        writer.writerow(stat)
print(f"CSV results also saved to {output_path_csv}")

import pandas as pd
print("Plotting results ...")
df = pd.read_csv(output_path_csv)

print(df.head())

if "model_name" in df.columns:
    df = df.drop(columns=["model_name", "input_text",	"reference_solution"	,"generated_text"])

df["num_threads"] = pd.to_numeric(df["num_threads"], errors='coerce')

grouped = df.groupby("num_threads", as_index=False).mean()
print("Grouped DataFrame:")
print(grouped)

plt.figure(figsize=(8,6))
plt.plot(grouped["num_threads"], grouped["elapsed_time_s"], marker='o', linestyle='-')
plt.xlabel("Number of Threads")
plt.ylabel("Average Elapsed Time (s)")
plt.title("Average Inference Time vs Number of Threads")
plt.grid(True)

plt.xticks([1, 2, 4, 8])
plt.xlim(1, 8)

plt.show()

print("Plots saved to:", OUTPUT_DIR)

df = pd.read_csv(output_path_csv)
print("Loaded DataFrame:")
print(df.head())
if "model_name" in df.columns:
    df = df.drop(columns=["model_name", "input_text",	"reference_solution"	,"generated_text"])

if 'num_threads' not in df.columns:
    raise ValueError("CSV中没有 'num_threads' 列，请检查实验数据。")

df["num_threads"] = pd.to_numeric(df["num_threads"], errors='coerce')

grouped = df.groupby("num_threads", as_index=False).mean()
print("Grouped DataFrame:")
print(grouped)

plt.figure(figsize=(8,6))
plt.plot(grouped["num_threads"], grouped["cpu_usage_before"], marker='o', linestyle='-', label="CPU Usage Before")
plt.plot(grouped["num_threads"], grouped["cpu_usage_after"], marker='x', linestyle='-', label="CPU Usage After")
plt.xlabel("Number of Threads")
plt.ylabel("CPU Usage (%)")
plt.title("Average CPU Usage vs Number of Threads")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8,6))
plt.plot(grouped["num_threads"], grouped["mem_usage_before"], marker='o', linestyle='-', label="Memory Usage Before")
plt.plot(grouped["num_threads"], grouped["mem_usage_after"], marker='x', linestyle='-', label="Memory Usage After")
plt.xlabel("Number of Threads")
plt.ylabel("Memory Usage (bytes)")
plt.title("Average Memory Usage vs Number of Threads")
plt.legend()
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt
import numpy as np

threads = np.array([1, 2, 4, 8])

plt.figure(figsize=(7,5))
for mv in model_variations:
    plt.plot(threads, inference_time[mv], marker='o', label=mv)

plt.title("Average Inference Time vs Number of Threads\nfor Different BERT Model Variations")
plt.xlabel("Number of Threads")
plt.ylabel("Average Elapsed Time (s)")
plt.xticks(threads)
plt.grid(True)
plt.legend(title="Model Variation")
plt.savefig("bert_inference_time.png", dpi=120)
plt.show()

plt.figure(figsize=(7,5))
for mv in model_variations:
    plt.plot(threads, cpu_diff[mv], marker='o', label=mv)

plt.title("Average CPU Usage Difference vs Number of Threads\nfor Different BERT Model Variations")
plt.xlabel("Number of Threads")
plt.ylabel("Average CPU Usage Difference (%)")
plt.xticks(threads)
plt.grid(True)
plt.legend(title="Model Variation")
plt.savefig("bert_cpu_diff.png", dpi=120)
plt.show()

plt.figure(figsize=(7,5))
for mv in model_variations:
    plt.plot(threads, mem_diff[mv], marker='o', label=mv)

plt.title("Average Memory Usage Difference vs Number of Threads\nfor Different BERT Model Variations")
plt.xlabel("Number of Threads")
plt.ylabel("Average Memory Usage Difference (bytes)")
plt.xticks(threads)
plt.grid(True)
plt.legend(title="Model Variation")
plt.savefig("bert_mem_diff.png", dpi=120)
plt.show()


data_for_boxplot = [base_times, pruned_times, quantized_times]

plt.figure(figsize=(7,5))
plt.boxplot(data_for_boxplot, labels=model_variations)
plt.title("Inference Time Distribution for Each BERT Model Variation")
plt.xlabel("Model Variation")
plt.ylabel("Inference Time (s)")
plt.grid(True)
plt.savefig("bert_inference_time_boxplot.png", dpi=120)
plt.show()

print("All 4 BERT figures have been saved locally:")
print("1) bert_inference_time.png")
print("2) bert_cpu_diff.png")
print("3) bert_mem_diff.png")
print("4) bert_inference_time_boxplot.png")

