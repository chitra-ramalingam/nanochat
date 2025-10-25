"""
Utilities for saving and loading model/optim/state checkpoints.
"""
import os
import re
import glob
import json
import logging
import torch

from nanochat.common import get_base_dir
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer
from nanochat.common import setup_default_logging
from transformers import AutoTokenizer, AutoModelForCausalLM


# Set up logging
setup_default_logging()
logger = logging.getLogger(__name__)
def log0(message):
    if int(os.environ.get('RANK', 0)) == 0:
        logger.info(message)

def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data):
    assert int(os.environ.get('RANK', 0)) == 0 # prevent footguns for now
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Save the model state (parameters)
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    torch.save(model_data, model_path)
    log0(f"Saved model file to: {model_path}")
    # Save the optimizer state (useful for SFT or any other fine-tuning)
    if optimizer_data is not None:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}.pt")
        torch.save(optimizer_data, optimizer_path)
        log0(f"Saved optimizer file to: {optimizer_path}")
    # Save the metadata dict as json
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "w") as f:
        json.dump(meta_data, f, indent=2)
    log0(f"Saved metadata file to: {meta_path}")


def load_checkpoint(checkpoint_dir, step, device, load_optimizer=False):
    # Load the model state
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    model_data = torch.load(model_path, map_location=device)
    # Load the optimizer state if requested
    optimizer_data = None
    if load_optimizer:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}.pt")
        optimizer_data = torch.load(optimizer_path, map_location=device)
    # Load the metadata
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r") as f:
        meta_data = json.load(f)
    return model_data, optimizer_data, meta_data


def build_model(checkpoint_dir, step, device, phase):
    """
    A bunch of repetitive code to build a model from a given checkpoint.
    Returns:
    - base model - uncompiled, not wrapped in DDP
    - tokenizer
    - meta data saved during base model training
    """
    assert phase in ["train", "eval"], f"Invalid phase: {phase}"
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)
    # Hack: fix torch compile issue, which prepends all keys with _orig_mod.
    model_data = {k.lstrip("_orig_mod."): v for k, v in model_data.items()}
    model_config_kwargs = meta_data["model_config"]
    log0(f"Building model with config: {model_config_kwargs}")
    model_config = GPTConfig(**model_config_kwargs)
    with torch.device("meta"):
        model = GPT(model_config)
    # Load the model state
    model.to_empty(device=device)
    model.init_weights() # note: this is dumb, but we need to init the rotary embeddings. TODO: fix model re-init
    model.load_state_dict(model_data, strict=True, assign=True)
    # Put the model in the right training phase / mode
    if phase == "eval":
        model.eval()
    else:
        model.train()
    # Load the Tokenizer
    tokenizer = get_tokenizer()
    # Sanity check: compatibility between model and tokenizer
    assert tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"]
    return model, tokenizer, meta_data


def find_largest_model(checkpoint_dir):
    # attempt to guess the model tag: take the biggest model available
    model_tags = [f for f in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, f))]
    if not model_tags:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    # 1) normally all model tags are of the form d<number>, try that first:
    candidates = []
    for model_tag in model_tags:
        match = re.match(r"d(\d+)", model_tag)
        if match:
            model_depth = int(match.group(1))
            candidates.append((model_depth, model_tag))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    # 2) if that failed, take the most recently updated model:
    model_tags.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
    return model_tags[0]


def find_last_step(checkpoint_dir):
    # Look into checkpoint_dir and find model_<step>.pt with the highest step
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    last_step = int(max(os.path.basename(f).split("_")[-1].split(".")[0] for f in checkpoint_files))
    return last_step


SPECIAL_TOKENS = [
    "<|user_start|>", "<|user_end|>",
    "<|assistant_start|>", "<|assistant_end|>",
    "<|system_start|>", "<|system_end|>",
]

def _adapt_tokenizer(tok):
    # Add NanoChat-style helpers that the code expects
    if not hasattr(tok, "get_bos_token_id"):
        setattr(tok, "get_bos_token_id", lambda: getattr(tok, "bos_token_id", None))
    if not hasattr(tok, "get_eos_token_id"):
        setattr(tok, "get_eos_token_id", lambda: getattr(tok, "eos_token_id", None))
    if not hasattr(tok, "get_vocab_size"):
        if hasattr(tok, "vocab_size"):
            setattr(tok, "get_vocab_size", lambda: tok.vocab_size)
        else:
            setattr(tok, "get_vocab_size", lambda: None)
    if not hasattr(tok, "encode_special"):
        setattr(tok, "encode_special", lambda s: [tok.convert_tokens_to_ids(s)])
    return tok


def load_model_from_hf(repo_id: str, device, phase="eval", dtype: str | None = None, **kwargs):
    if dtype == "bfloat16": torch_dtype = torch.bfloat16
    elif dtype == "float32": torch_dtype = torch.float32
    else:
        torch_dtype = torch.float16 if device.type == "cuda" else (torch.bfloat16 if device.type == "mps" else torch.float32)

    tok = AutoTokenizer.from_pretrained(repo_id, use_fast=True)

    to_add = [t for t in SPECIAL_TOKENS if tok.convert_tokens_to_ids(t) in (None, tok.unk_token_id, -1)]
    if to_add:
        tok.add_special_tokens({"additional_special_tokens": to_add})

    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map="auto" if device.type in ("cuda", "mps") else None
    )

    # ---- HF config -> NanoChat config shim --------------------------------
    cfg = model.config
    if not hasattr(cfg, "n_head") and hasattr(cfg, "num_attention_heads"):
        cfg.n_head = cfg.num_attention_heads
    if not hasattr(cfg, "n_kv_head"):
        if hasattr(cfg, "num_key_value_heads") and cfg.num_key_value_heads is not None:
            cfg.n_kv_head = cfg.num_key_value_heads
        else:
            cfg.n_kv_head = getattr(cfg, "num_attention_heads", None)  # non-GQA fallback
    if not hasattr(cfg, "n_embd") and hasattr(cfg, "hidden_size"):
        cfg.n_embd = cfg.hidden_size
    if not hasattr(cfg, "n_layer") and hasattr(cfg, "num_hidden_layers"):
        cfg.n_layer = cfg.num_hidden_layers
    # -----------------------------------------------------------------------

    # --- Adapters so Nanochat's engine works with HF models ----------------
    if not hasattr(model, "get_device"):
        def _get_device():
            try:
                return next(model.parameters()).device
            except StopIteration:
                return torch.device("cpu")
        setattr(model, "get_device", _get_device)

    if not hasattr(model, "max_seq_len"):
        max_len = getattr(getattr(model, "config", None), "max_position_embeddings", None)
        if max_len is None:
            try:
                max_len = tok.model_max_length
                if isinstance(max_len, int) and max_len > 32768:
                    max_len = 4096
            except Exception:
                max_len = 4096
        setattr(model, "max_seq_len", max_len)
    # -----------------------------------------------------------------------

    if to_add:
        model.resize_token_embeddings(len(tok))

    if device.type == "cpu":
        model.to(device)
    model.eval() if phase == "eval" else model.train()

    tok = _adapt_tokenizer(tok)

    meta = {"source": "hf", "repo_id": repo_id, "model_config": {"vocab_size": tok.get_vocab_size()}}
    return model, tok, meta

# -----------------------------------------------------------------------------
# convenience functions that take into account nanochat's directory structure

def load_model_from_dir(checkpoints_dir, device, phase, model_tag=None, step=None):
    if model_tag is None:
        # guess the model tag by defaulting to the largest model
        model_tag = find_largest_model(checkpoints_dir)
        log0(f"No model tag provided, guessing model tag: {model_tag}")
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if step is None:
        # guess the step by defaulting to the last step
        step = find_last_step(checkpoint_dir)
    assert step is not None, f"No checkpoints found in {checkpoint_dir}"
    # build the model
    log0(f"Loading model from {checkpoint_dir} with step {step}")
    model, tokenizer, meta_data = build_model(checkpoint_dir, step, device, phase)
    return model, tokenizer, meta_data

def load_model(source, device, phase="eval", model_tag=None, step=None):
    source = (source or "").lower()
    if source in {"base", "mid", "sft", "rl"}:
        model_dir = {
            "base": "base_checkpoints",
            "mid":  "mid_checkpoints",
            "sft":  "chatsft_checkpoints",
            "rl":   "chatrl_checkpoints",
        }[source]
        base_dir = get_base_dir()
        checkpoints_dir = os.path.join(base_dir, model_dir)
        return load_model_from_dir(checkpoints_dir, device, phase, model_tag=model_tag, step=step)

    elif source == "hf":
        if not model_tag:
            raise ValueError("When source='hf', pass -g/--model-tag as an HF repo id (e.g. microsoft/phi-3-mini-4k-instruct)")
        return load_model_from_hf(model_tag, device, phase=phase)

    else:
        raise KeyError(source)
