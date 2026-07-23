from __future__ import annotations

from pathlib import Path

import torch

from utils.clip.simple_tokenizer import SimpleTokenizer
from utils.clip.text_encoder import Text_encoder


_TOKENIZER = SimpleTokenizer()


def tokenize(texts: list[str], context_length: int = 77, truncate: bool = True) -> torch.Tensor:
    sot_token = _TOKENIZER.encoder["<|startoftext|>"]
    eot_token = _TOKENIZER.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _TOKENIZER.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)
    for index, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if not truncate:
                raise RuntimeError(f"Input text is too long for CLIP context length: {texts[index]}")
            tokens = tokens[:context_length]
            tokens[-1] = eot_token
        result[index, : len(tokens)] = torch.tensor(tokens, dtype=torch.int)
    return result


def load_text_encoder(checkpoint_path: str | Path, device: str | torch.device = "cpu"):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"CLIP checkpoint not found: {checkpoint_path}")

    pretrained = torch.jit.load(str(checkpoint_path), map_location=device).state_dict()
    embed_dim = pretrained["text_projection"].shape[1]
    context_length = pretrained["positional_embedding"].shape[0]
    vocab_size = pretrained["token_embedding.weight"].shape[0]
    transformer_width = pretrained["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = 12

    model = Text_encoder(
        embed_dim=embed_dim,
        context_length=context_length,
        vocab_size=vocab_size,
        transformer_width=transformer_width,
        transformer_heads=transformer_heads,
        transformer_layers=transformer_layers,
    )
    model_state = model.state_dict()
    text_state = {
        key: value
        for key, value in pretrained.items()
        if key in model_state and "visual" not in key.split(".")
    }
    for key in ["input_resolution", "context_length", "vocab_size"]:
        text_state.pop(key, None)
    model_state.update(text_state)
    model.load_state_dict(model_state)
    model.to(device).eval()
    return model, int(embed_dim), int(context_length)


@torch.no_grad()
def encode_texts(texts: list[str], checkpoint_path: str | Path, device: str | torch.device = "cpu") -> torch.Tensor:
    model, _, context_length = load_text_encoder(checkpoint_path, device)
    tokens = tokenize(texts, context_length=context_length, truncate=True).to(device)
    encoded = model(tokens).float()
    return encoded / encoded.norm(dim=1, keepdim=True).clamp_min(1e-12)
