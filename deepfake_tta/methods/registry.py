from __future__ import annotations

from deepfake_tta.methods.bca import BCA, BCAConfig
from deepfake_tta.methods.boost_adapter import BoostAdapter, BoostAdapterConfig
from deepfake_tta.methods.compact_cache_adapter import CompactCacheAdapter, CompactCacheAdapterConfig
from deepfake_tta.methods.crg import CRG, CRGConfig
from deepfake_tta.methods.dmn import DMN, DMNConfig
from deepfake_tta.methods.dota import DOTA, DOTAConfig
from deepfake_tta.methods.dpe import DPE, DPEConfig
from deepfake_tta.methods.dynaprompt import DynaPrompt, DynaPromptConfig
from deepfake_tta.methods.freetta import FreeTTA, FreeTTAConfig
from deepfake_tta.methods.freetta_linear_ensemble import (
    FreeTTALinearEnsemble,
    FreeTTALinearEnsembleConfig,
)
from deepfake_tta.methods.online_confident_cache_adapter import (
    OnlineConfidentCacheAdapter,
    OnlineConfidentCacheAdapterConfig,
)
from deepfake_tta.methods.prototype_linear_tta import PrototypeLinearTTA, PrototypeLinearTTAConfig
from deepfake_tta.methods.tip_adapter import TipAdapter, TipAdapterConfig

AVAILABLE_TTA_METHODS = (
    "tip_adapter",
    "boost_adapter",
    "compact_cache_adapter",
    "online_confident_cache_adapter",
    "crg",
    "dmn",
    "dpe",
    "dota",
    "freetta",
    "freetta_linear_ensemble",
    "bca",
    "dynaprompt",
    "prototype_linear_tta",
)


def create_tta_method(
    name: str,
    *,
    alpha: float = 0.5,
    beta: float = 5.5,
    test_batch_size: int = 512,
    cache_batch_size: int = 8192,
    method_cache_dir: str | None = None,
):
    if name == "tip_adapter":
        return TipAdapter(
            TipAdapterConfig(
                alpha=alpha,
                beta=beta,
                test_batch_size=test_batch_size,
                cache_batch_size=cache_batch_size,
            )
        )
    if name == "boost_adapter":
        return BoostAdapter(
            BoostAdapterConfig(
                alpha=alpha,
                beta=beta,
                batch_size=test_batch_size,
                cache_batch_size=cache_batch_size,
            )
        )
    if name == "compact_cache_adapter":
        return CompactCacheAdapter(
            CompactCacheAdapterConfig(
                alpha=alpha,
                beta=beta,
                batch_size=test_batch_size,
                cache_ratio=0.5,
            )
        )
    if name == "online_confident_cache_adapter":
        cache_path = None
        if method_cache_dir:
            cache_path = f"{method_cache_dir}/online_confident_source_cache_ratio01.pt"
        return OnlineConfidentCacheAdapter(
            OnlineConfidentCacheAdapterConfig(
                beta=beta,
                batch_size=test_batch_size,
                source_cache_ratio=0.1,
                alpha_source=0.2,
                alpha_dynamic=0.25,
                add_threshold=0.9,
                max_entropy=0.35,
                entropy_score_weight=0.25,
                use_threshold=0.85,
                agreement_only=True,
                max_dynamic_per_class=2048,
                cache_path=cache_path,
                save_cache=True,
            )
        )
    if name == "crg":
        return CRG(CRGConfig(beta=beta, batch_size=test_batch_size))
    if name == "dmn":
        return DMN(DMNConfig(beta=beta, batch_size=test_batch_size))
    if name == "dpe":
        return DPE(DPEConfig(batch_size=test_batch_size))
    if name == "bca":
        return BCA(BCAConfig(batch_size=test_batch_size))
    if name == "dota":
        return DOTA(DOTAConfig(batch_size=test_batch_size))
    if name == "freetta":
        return FreeTTA(FreeTTAConfig(batch_size=test_batch_size))
    if name == "freetta_linear_ensemble":
        return FreeTTALinearEnsemble(
            FreeTTALinearEnsembleConfig(
                batch_size=test_batch_size,
                linear_weight=0.3,
                freetta_weight=0.7,
            )
        )
    if name == "dynaprompt":
        return DynaPrompt(DynaPromptConfig(batch_size=test_batch_size))
    if name == "prototype_linear_tta":
        return PrototypeLinearTTA(PrototypeLinearTTAConfig(batch_size=test_batch_size))
    raise ValueError(f"Unknown TTA method: {name}")
