from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class SpectralMAEBackbone(nn.Module):
    def __init__(self, input_bands: int, feature_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(int(input_bands), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(feature_dim)),
            nn.LayerNorm(int(feature_dim)),
        )
        self.decoder = nn.Sequential(
            nn.Linear(int(feature_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(input_bands)),
        )
        self.input_bands = int(input_bands)
        self.feature_dim = int(feature_dim)

    def forward(self, spectra: torch.Tensor) -> torch.Tensor:
        if spectra.ndim != 2:
            raise ValueError("spectra must have shape [N, Bands]")
        return self.encoder(spectra)

    def reconstruct(self, spectra: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.forward(spectra))

    def freeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False


class SpectralMorphologyBackbone(nn.Module):
    """Stable MLP features plus local slope/curvature residuals."""

    def __init__(self, input_bands: int, feature_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.base = SpectralMAEBackbone(input_bands, feature_dim, hidden_dim)
        self.local = nn.ModuleList([
            nn.Conv1d(3, 3, kernel, padding=kernel // 2, groups=3)
            for kernel in (3, 5, 9)
        ])
        self.projection = nn.Sequential(nn.Linear(9, feature_dim), nn.LayerNorm(feature_dim))
        self.gate = nn.Parameter(torch.zeros(()))
        self.input_bands = int(input_bands)
        self.feature_dim = int(feature_dim)

    def _morphology(self, spectra: torch.Tensor) -> torch.Tensor:
        slope = torch.diff(spectra, dim=1, prepend=spectra[:, :1])
        curvature = torch.diff(slope, dim=1, prepend=slope[:, :1])
        signals = torch.stack((spectra, slope, curvature), dim=1)
        pooled = [torch.nn.functional.gelu(layer(signals)).mean(dim=2) for layer in self.local]
        return self.projection(torch.cat(pooled, dim=1))

    def morphology_features(self, spectra: torch.Tensor) -> torch.Tensor:
        return self._morphology(spectra)

    def forward(self, spectra: torch.Tensor) -> torch.Tensor:
        if spectra.ndim != 2 or spectra.shape[1] != self.input_bands:
            raise ValueError(f"spectra must have shape [N, {self.input_bands}]")
        return self.base(spectra) + torch.tanh(self.gate) * self._morphology(spectra)

    def reconstruct(self, spectra: torch.Tensor) -> torch.Tensor:
        return self.base.decoder(self.forward(spectra))

    def freeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False


class PhysicsInvariantBackbone(nn.Module):
    def __init__(self, input_bands: int, feature_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.base = SpectralMorphologyBackbone(input_bands, feature_dim, hidden_dim)
        self.local = nn.ModuleList([
            nn.Conv1d(3, 3, kernel, padding=kernel // 2, groups=3)
            for kernel in (3, 7, 15)
        ])
        self.projection = nn.Sequential(nn.Linear(9, feature_dim), nn.LayerNorm(feature_dim))
        self.gate = nn.Parameter(torch.zeros(()))
        self.input_bands = int(input_bands)
        self.feature_dim = int(feature_dim)

    def _views(self, spectra: torch.Tensor) -> torch.Tensor:
        mean = spectra.mean(1, keepdim=True)
        snv = (spectra - mean) / spectra.std(1, keepdim=True, unbiased=False).clamp_min(1e-4)
        trend = torch.nn.functional.avg_pool1d(
            spectra[:, None], kernel_size=15, stride=1, padding=7,
        ).squeeze(1).clamp_min(1e-4)
        continuum = spectra / trend
        derivative = torch.diff(spectra, dim=1, prepend=spectra[:, :1])
        derivative = derivative / derivative.abs().mean(1, keepdim=True).clamp_min(1e-4)
        return torch.stack((snv, continuum, derivative), dim=1)

    def physics_features(self, spectra: torch.Tensor) -> torch.Tensor:
        pooled = [torch.nn.functional.gelu(layer(self._views(spectra))).mean(2) for layer in self.local]
        return self.projection(torch.cat(pooled, dim=1))

    def forward(self, spectra: torch.Tensor) -> torch.Tensor:
        return self.base(spectra) + torch.tanh(self.gate) * self.physics_features(spectra)

    def reconstruct(self, spectra: torch.Tensor) -> torch.Tensor:
        return self.base.base.decoder(self.forward(spectra))

    def freeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False


class WavelengthTransformerBackbone(nn.Module):
    def __init__(
        self, input_bands: int, feature_dim: int = 64, hidden_dim: int = 256,
        num_layers: int = 2, num_heads: int = 4,
    ):
        super().__init__()
        if int(feature_dim) % int(num_heads):
            raise ValueError("feature_dim must be divisible by num_heads")
        self.value_projection = nn.Linear(1, int(feature_dim))
        self.wavelength_projection = nn.Sequential(nn.Linear(1, int(feature_dim)), nn.GELU())
        self.cls_token = nn.Parameter(torch.zeros(1, 1, int(feature_dim)))
        layer = nn.TransformerEncoderLayer(
            d_model=int(feature_dim), nhead=int(num_heads), dim_feedforward=int(hidden_dim),
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers), enable_nested_tensor=False)
        self.norm = nn.LayerNorm(int(feature_dim))
        self.band_decoder = nn.Linear(int(feature_dim), 1)
        self.register_buffer("wavelengths", torch.linspace(0.0, 1.0, int(input_bands)).reshape(1, -1, 1))
        self.input_bands = int(input_bands)
        self.feature_dim = int(feature_dim)

    def _tokens(self, spectra: torch.Tensor) -> torch.Tensor:
        if spectra.ndim != 2 or spectra.shape[1] != self.input_bands:
            raise ValueError(f"spectra must have shape [N, {self.input_bands}]")
        bands = self.value_projection(spectra.unsqueeze(-1)) + self.wavelength_projection(self.wavelengths)
        cls = self.cls_token.expand(spectra.shape[0], -1, -1)
        return self.norm(self.encoder(torch.cat([cls, bands], dim=1)))

    def forward(self, spectra: torch.Tensor) -> torch.Tensor:
        return self._tokens(spectra)[:, 0]

    def reconstruct(self, spectra: torch.Tensor) -> torch.Tensor:
        return self.band_decoder(self._tokens(spectra)[:, 1:]).squeeze(-1)

    def freeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False


def load_backbone_checkpoint(backbone: nn.Module, checkpoint_path: str | Path) -> None:
    try:
        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
    backbone.load_state_dict(checkpoint.get("backbone", checkpoint))


def build_backbone(input_bands: int, config: dict) -> nn.Module:
    model_config = config.get("model", {})
    backbone_name = str(model_config.get("backbone", "spectral_mae")).lower()
    spectral_config = config.get("spectral_mae", {})
    parameters = {
        "input_bands": input_bands,
        "feature_dim": int(model_config.get("feature_dim", spectral_config.get("feature_dim", 64))),
        "hidden_dim": int(spectral_config.get("hidden_dim", 256)),
    }
    if backbone_name == "spectral_mae":
        backbone = SpectralMAEBackbone(**parameters)
    elif backbone_name == "spectral_morphology":
        backbone = SpectralMorphologyBackbone(**parameters)
    elif backbone_name == "physics_invariant":
        backbone = PhysicsInvariantBackbone(**parameters)
    elif backbone_name == "wavelength_transformer":
        backbone = WavelengthTransformerBackbone(
            **parameters, num_layers=int(spectral_config.get("num_layers", 2)),
            num_heads=int(spectral_config.get("num_heads", 4)),
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")
    checkpoint = model_config.get("pretrained_backbone")
    if checkpoint:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path(__file__).resolve().parents[1] / checkpoint_path
        if checkpoint_path.exists():
            load_backbone_checkpoint(backbone, checkpoint_path)
            print(f"[PhyRC-GZSL] loaded pretrained backbone: {checkpoint_path}")
        else:
            print(f"[PhyRC-GZSL][WARN] pretrained backbone not found: {checkpoint_path}")
    if model_config.get("freeze_backbone", True):
        backbone.freeze()
    return backbone
