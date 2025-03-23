import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_CONFIGS = {
    "gpt2": {
        "pretrained_name": "gpt2",
        "revision": "main",
        "max_length": 1024
    },
    # "gpt2-large": {
    #     "pretrained_name": "gpt2-large",
    #     "revision": "main",
    #     "max_length": 1024
    # },
    # "opt-125m": {
    #     "pretrained_name": "facebook/opt-125m",
    #     "revision": "main",
    #     "max_length": 512
    # },
    # "TinyLlama-1.1B-Chat-v1.0": {
    #     "pretrained_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    #     "revision": "step3000",
    #     "max_length": 512
    # },
    # "Qwen/Qwen2.5-7B": {
    #     "pretrained_name": "Qwen/Qwen2.5-7B",
    #     "revision": "main",
    #     "max_length": 2048
    # }
}


def load_model_and_tokenizer(model_name):
    """Load model with configurable precision and memory optimization"""
    config = MODEL_CONFIGS[model_name]

    tokenizer = AutoTokenizer.from_pretrained(
        config["pretrained_name"],
        revision=config["revision"]
    )

    model = AutoModelForCausalLM.from_pretrained(
        config["pretrained_name"],
        revision=config["revision"],
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True
    )

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, config["max_length"]