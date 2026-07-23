from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch


SENSOR_TOKEN_KEYS = (
    "identity", "reflectance_level", "visible_nir_slope",
    "curvature_transition", "red_edge", "intra_class_variation",
)


def validate_sensor_descriptions(descriptions: list[str]) -> list[str]:
    if len(descriptions) != len(SENSOR_TOKEN_KEYS):
        raise ValueError("sensor descriptions must contain exactly six slots")
    forbidden = (
        "swir", "shortwave infrared", "thermal", "spatial", "texture", "geometry",
        "roof", "road network", "canopy height", "micrometer", "1.4 ", "1.5 ",
        "1.7 ", "1.9 ", "2.2 ", "2.3 ", "2.4 ", "2.5 ",
    )
    for text in descriptions:
        lowered = text.lower()
        if any(term in lowered for term in forbidden) or any(ord(char) > 127 for char in text):
            raise ValueError(f"unobservable or malformed sensor description: {text}")
    return descriptions


def _stable_vector(text: str, dim: int) -> np.ndarray:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False)
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=dim).astype(np.float32)
    vector /= max(float(np.linalg.norm(vector)), 1e-12)
    return vector


class SemanticConditionEncoder:
    def __init__(self, config: dict, project_root: Path):
        self.config = config
        self.project_root = project_root
        self.text_cfg = config.get("text", {})
        self.condition_dim = int(self.text_cfg.get("condition_dim", 768))
        cache_rel = self.text_cfg.get("cache_file", "data/processed/text_descriptions.json")
        self.cache_path = project_root / "phyrc_gzsl" / cache_rel
        self._clip_model = None
        self._clip_context_length = None
        self._clip_device = None

    def _load_cache(self) -> dict:
        if not self.cache_path.exists():
            return {}
        return json.loads(self.cache_path.read_text(encoding="utf-8"))

    def _save_cache(self, cache: dict) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")

    def _fallback_descriptions(self, class_name: str) -> list[str]:
        if self.text_cfg.get("semantic_mode") == "sensor_tokens":
            return [
                f"A single hyperspectral pixel of {class_name}.",
                f"The reflectance level of {class_name} is described only within 430-860 nm.",
                f"The visible to near infrared slope of {class_name} may distinguish it from other materials.",
                f"Broad spectral curvature of {class_name} is considered only within the observed bands.",
                f"Vegetation red-edge behavior is included only when physically applicable to {class_name}.",
                f"Illumination moisture and material mixing may vary the observed spectrum of {class_name}.",
            ]
        return [class_name for _ in range(int(self.text_cfg.get("descriptions_per_class", 5)))]

    def _deepseek_descriptions(self, class_name: str) -> list[str]:
        api_key = self.text_cfg.get("api_key") or os.getenv(self.text_cfg.get("api_key_env", "DEEPSEEK_API_KEY"))
        if not api_key:
            raise RuntimeError("DeepSeek API key is not configured")
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=self.text_cfg.get("base_url", "https://api.deepseek.com"))
        count = len(SENSOR_TOKEN_KEYS) if self.text_cfg.get("semantic_mode") == "sensor_tokens" else int(self.text_cfg.get("descriptions_per_class", 5))
        if self.text_cfg.get("semantic_mode") == "sensor_tokens":
            prompt = f"""
You are writing conditioning text for a single-pixel hyperspectral generative model.
The class is "{class_name}" and the only sensor measurements are 103 reflectance bands from 430 to 860 nm.

Return one valid JSON object with exactly these six string keys in this exact order:
identity, reflectance_level, visible_nir_slope, curvature_transition, red_edge, intra_class_variation.

Rules:
- Each value must be one short English sentence of at most 24 words and contain ASCII characters only.
- Describe only spectral behavior observable from one 430-860 nm pixel.
- Never mention wavelengths outside 430-860 nm, SWIR, thermal properties, spatial context, texture, geometry, object size, height, or land-use patterns.
- Do not invent narrow absorption bands. Prefer cautious qualitative statements about reflectance level, slope, broad curvature, and transitions.
- identity: identify the class as a single hyperspectral pixel; add no spatial information.
- reflectance_level: state the expected low, medium, or high reflectance within the observed range, allowing variability when necessary.
- visible_nir_slope: describe the broad direction of change from visible wavelengths toward 860 nm.
- curvature_transition: describe only broad curve shape or transitions visible inside the sensor range.
- red_edge: describe red-edge behavior for vegetation; explicitly state its absence or non-diagnostic nature for non-vegetation.
- intra_class_variation: mention only spectral variation caused by illumination, moisture, aging, composition, or sub-pixel mixing.
- If a property is uncertain, say it is variable rather than fabricating a precise feature.

Output JSON only. Do not use Markdown.
""".strip()
        elif self.text_cfg.get("semantic_mode") == "sensor_aligned":
            low, high = self.text_cfg.get("wavelength_nm", [430, 860])
            prompt = (
                f"Generate {count} distinct English descriptions of the single-pixel spectral signature of "
                f"'{class_name}' observable by a hyperspectral sensor covering only {low}-{high} nm. "
                "Describe only observable reflectance level, visible-band slope, spectral curvature, red-edge "
                "behavior when applicable, and absorption or transition features inside this wavelength range. "
                "Do not mention SWIR wavelengths, spatial shape, texture, object geometry, temperature, emissivity, "
                "or properties that cannot be inferred from one pixel. Return one description per line without numbering."
            )
        else:
            prompt = (
                f"Generate {count} fine-grained English descriptions for hyperspectral remote sensing class "
                f"'{class_name}'. Each description must mention spectral, spatial, and physical properties. "
                "Return one description per line without numbering."
            )
        response = client.chat.completions.create(
            model=self.text_cfg.get("model", "deepseek-v4-flash"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        content = response.choices[0].message.content or ""
        if self.text_cfg.get("semantic_mode") == "sensor_tokens":
            content = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            payload = json.loads(content)
            lines = validate_sensor_descriptions([str(payload[key]).strip() for key in SENSOR_TOKEN_KEYS])
        else:
            lines = [line.strip(" -\t0123456789.") for line in content.splitlines() if line.strip()]
        return (lines[:count] or self._fallback_descriptions(class_name))[:count]

    def get_descriptions(self, class_names: dict[int, str]) -> dict[str, list[str]]:
        cache = self._load_cache()
        changed = False
        for class_id, class_name in class_names.items():
            key = str(class_id)
            if not self.text_cfg.get("refresh_cache", False) and key in cache and cache[key].get("descriptions"):
                continue
            try:
                descriptions = self._deepseek_descriptions(class_name)
            except Exception as exc:
                print(f"[PhyRC-GZSL][WARN] DeepSeek description generation failed for {class_name}: {exc}")
                descriptions = self._fallback_descriptions(class_name)
            cache[key] = {"class_name": class_name, "descriptions": descriptions}
            changed = True
        if changed:
            self._save_cache(cache)
        return {key: value["descriptions"] for key, value in cache.items()}

    def _encode_with_clip(self, descriptions: list[str]) -> torch.Tensor:
        return self._encode_with_clip_tokens(descriptions).mean(dim=0)

    def _encode_with_clip_tokens(self, descriptions: list[str]) -> torch.Tensor:
        checkpoint = self.text_cfg.get("clip_pretrained") or "extract_text/ViT-L-14.pt"
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = self.project_root / checkpoint_path
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            from phyrc_gzsl.utils.clip.local_text import load_text_encoder, tokenize

            if self._clip_model is None or self._clip_device != device:
                self._clip_model, _, self._clip_context_length = load_text_encoder(checkpoint_path, device=device)
                self._clip_device = device
            tokens = tokenize(descriptions, context_length=self._clip_context_length, truncate=True).to(device)
            with torch.no_grad():
                encoded = self._clip_model(tokens).float()
                encoded = encoded / encoded.norm(dim=1, keepdim=True).clamp_min(1e-12)
                return encoded.cpu()
        except Exception as exc:
            print(f"[PhyRC-GZSL][WARN] local CLIP text encoding failed: {exc}")
            vectors = np.stack([_stable_vector(text, self.condition_dim) for text in descriptions], axis=0)
            return torch.from_numpy(vectors)

    def encode_class_tokens(self, class_names: dict[int, str]) -> dict[int, torch.Tensor]:
        descriptions = self.get_descriptions(class_names)
        return {
            int(class_id): self._encode_with_clip_tokens(
                descriptions.get(str(class_id), self._fallback_descriptions(class_name))
            ).float()
            for class_id, class_name in class_names.items()
        }

    def encode_classes(self, class_names: dict[int, str]) -> dict[int, torch.Tensor]:
        token_map = self.encode_class_tokens(class_names)
        embeddings: dict[int, torch.Tensor] = {}
        for class_id, tokens in token_map.items():
            vector = tokens.mean(dim=0).float()
            vector = vector / vector.norm().clamp_min(1e-12)
            embeddings[int(class_id)] = vector
        return embeddings


def stack_conditions(embeddings: dict[int, torch.Tensor], class_ids: list[int]) -> torch.Tensor:
    return torch.stack([embeddings[int(class_id)] for class_id in class_ids], dim=0)
