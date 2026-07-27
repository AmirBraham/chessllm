"""Load Qwen3-0.6B from Hugging Face and generate completions in batches.

Batching is the point. Generation is the bottleneck in every eval and every
GRPO rollout, and running prompts one at a time leaves the GPU mostly idle.
"""

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"  # post-trained ("thinking") model, not -Base


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_vllm(model_id, max_model_len=4096, gpu_fraction=0.90):
    """vLLM backend: continuous batching, so a slow sequence stops holding up
    the rest of the batch.

    Opt-in because vLLM pins torch exactly (0.26 wants 2.11.0), which replaces
    the image's driver-matched build. If the replacement is a cu13 wheel on a
    driver that only speaks 12.x, torch falls back to CPU without erroring --
    run gpucheck.py after installing it.
    """
    from vllm import LLM

    tok = AutoTokenizer.from_pretrained(model_id)
    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_fraction,
        max_model_len=max_model_len,
    )
    return llm, tok


def load(model_id=MODEL_ID, device=None, backend="hf"):
    if backend == "vllm":
        return load_vllm(model_id)

    device = device or get_device()

    tok = AutoTokenizer.from_pretrained(model_id)
    # Decoder-only models must pad on the LEFT for batched generation. With
    # right padding the pad tokens sit between the prompt and the first
    # generated token, and the output is quietly wrong rather than an error.
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    model.to(device).eval()
    return model, tok


def generate(model, tok, prompts, max_new_tokens=512, think=True,
             temperature=None, batch_size=16, progress=True):
    """Generate one completion per prompt. Greedy unless `temperature` is set.

    Prompts run in batches of `batch_size`: one oversized batch runs out of
    memory, one prompt at a time leaves the GPU idle.

    Returns (texts, lengths). `lengths` is the number of tokens generated,
    which is how the caller detects truncation: a completion that used the
    entire budget never reached its answer, so it has no answer to score.
    """
    def as_chat(prompt):
        return tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=think,
        )

    # vLLM schedules its own batches, so batch_size is irrelevant there. Hand
    # it everything at once and let continuous batching do the work: a
    # sequence that finishes early frees its slot immediately instead of
    # waiting for the slowest one in a fixed batch.
    if type(model).__name__ == "LLM":
        from vllm import SamplingParams

        outputs = model.generate(
            [as_chat(prompt) for prompt in prompts],
            SamplingParams(
                max_tokens=max_new_tokens,
                temperature=0.0 if temperature is None else temperature,
            ),
        )
        return (
            [o.outputs[0].text for o in outputs],
            [len(o.outputs[0].token_ids) for o in outputs],
        )

    texts, lengths = [], []
    starts = range(0, len(prompts), batch_size)

    for start in tqdm(starts, unit="batch", disable=not progress):
        chunk = prompts[start:start + batch_size]
        chats = [as_chat(prompt) for prompt in chunk]

        batch = tok(chats, return_tensors="pt", padding=True).to(model.device)
        prompt_len = batch["input_ids"].shape[1]

        with torch.no_grad():
            out = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=temperature is not None,
                temperature=temperature,
                pad_token_id=tok.pad_token_id,
            )

        new = out[:, prompt_len:]
        texts.extend(tok.batch_decode(new, skip_special_tokens=True))
        lengths.extend(int((row != tok.pad_token_id).sum()) for row in new)

    return texts, lengths
