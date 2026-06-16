from __future__ import annotations

from deepfake_tta.methods.adaptive_dota_bca import AdaptiveDotaBca, AdaptiveDotaBcaConfig
from deepfake_tta.methods.bca import BCA, BCAConfig
from deepfake_tta.methods.boost_adapter import BoostAdapter, BoostAdapterConfig
from deepfake_tta.methods.compact_cache_adapter import CompactCacheAdapter, CompactCacheAdapterConfig
from deepfake_tta.methods.crg import CRG, CRGConfig
from deepfake_tta.methods.dmn import DMN, DMNConfig
from deepfake_tta.methods.dota import DOTA, DOTAConfig
from deepfake_tta.methods.dpe import DPE, DPEConfig
from deepfake_tta.methods.etta import ETTA, ETTAConfig
from deepfake_tta.methods.dynaprompt import DynaPrompt, DynaPromptConfig
from deepfake_tta.methods.freetta import FreeTTA, FreeTTAConfig
from deepfake_tta.methods.freetta_linear_ensemble import (
    FreeTTALinearEnsemble,
    FreeTTALinearEnsembleConfig,
)
from deepfake_tta.methods.gda import GDA, GDAConfig
from deepfake_tta.methods.lightweight_wrappers import (
    LinearEnsembleConfig,
    LinearEnsembleWrapper,
    PriorBalancedConfig,
    PriorBalancedWrapper,
)
from deepfake_tta.methods.online_confident_cache_adapter import (
    OnlineConfidentCacheAdapter,
    OnlineConfidentCacheAdapterConfig,
)
from deepfake_tta.methods.prototype_linear_tta import PrototypeLinearTTA, PrototypeLinearTTAConfig
from deepfake_tta.methods.tda import TDA, TDAConfig
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
    "tda",
    "gda",
    "etta",
    "freetta",
    "freetta_linear_ensemble",
    "freetta_balanced",
    "freetta_linear_ensemble_balanced",
    "bca_balanced",
    "bca_linear_ensemble_balanced",
    "dpe_balanced",
    "dpe_linear_ensemble_balanced",
    "dota_balanced",
    "dota_linear_ensemble_balanced",
    "adaptive_dota_bca",
    "prototype_linear_tta_balanced",
    "online_cache_10_balanced",
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
    def prior_balanced(inner, wrapper_name: str, strength: float = 0.7):
        return PriorBalancedWrapper(
            inner,
            name=wrapper_name,
            config=PriorBalancedConfig(
                batch_size=test_batch_size,
                target_prior=(0.5, 0.5),
                strength=strength,
            ),
        )

    def linear_ensemble(inner, wrapper_name: str, linear_weight: float = 0.3, inner_weight: float = 0.7):
        return LinearEnsembleWrapper(
            inner,
            name=wrapper_name,
            config=LinearEnsembleConfig(
                batch_size=test_batch_size,
                linear_weight=linear_weight,
                inner_weight=inner_weight,
            ),
        )

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
    if name == "adaptive_dota_bca":
        return AdaptiveDotaBca(AdaptiveDotaBcaConfig(batch_size=test_batch_size))
    if name == "tda":
        return TDA(TDAConfig(batch_size=test_batch_size))
    if name == "gda":
        return GDA(GDAConfig(batch_size=test_batch_size))
    if name == "etta":
        return ETTA(ETTAConfig(batch_size=test_batch_size))
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
    if name == "freetta_balanced":
        return prior_balanced(
            FreeTTA(FreeTTAConfig(batch_size=test_batch_size, base_weight=0.35)),
            "freetta_balanced",
            strength=0.8,
        )
    if name == "freetta_linear_ensemble_balanced":
        return prior_balanced(
            FreeTTALinearEnsemble(
                FreeTTALinearEnsembleConfig(
                    batch_size=test_batch_size,
                    linear_weight=0.25,
                    freetta_weight=0.75,
                )
            ),
            "freetta_linear_ensemble_balanced",
            strength=0.7,
        )
    if name == "dynaprompt":
        return DynaPrompt(DynaPromptConfig(batch_size=test_batch_size))
    if name == "prototype_linear_tta":
        return PrototypeLinearTTA(PrototypeLinearTTAConfig(batch_size=test_batch_size))
    if name == "bca_balanced":
        return prior_balanced(
            BCA(
                BCAConfig(
                    batch_size=test_batch_size,
                    base_weight=0.6,
                    prior_momentum=0.98,
                    prototype_momentum=0.99,
                )
            ),
            "bca_balanced",
            strength=0.7,
        )
    if name == "bca_linear_ensemble_balanced":
        return prior_balanced(
            linear_ensemble(
                BCA(
                    BCAConfig(
                        batch_size=test_batch_size,
                        base_weight=0.55,
                        prior_momentum=0.98,
                        prototype_momentum=0.99,
                    )
                ),
                "bca_linear_ensemble",
                linear_weight=0.35,
                inner_weight=0.65,
            ),
            "bca_linear_ensemble_balanced",
            strength=0.7,
        )
    if name == "dpe_balanced":
        return prior_balanced(
            DPE(DPEConfig(batch_size=test_batch_size, base_weight=0.55, visual_momentum=0.98)),
            "dpe_balanced",
            strength=0.7,
        )
    if name == "dpe_linear_ensemble_balanced":
        return prior_balanced(
            linear_ensemble(
                DPE(DPEConfig(batch_size=test_batch_size, base_weight=0.55, visual_momentum=0.98)),
                "dpe_linear_ensemble",
                linear_weight=0.35,
                inner_weight=0.65,
            ),
            "dpe_linear_ensemble_balanced",
            strength=0.7,
        )
    if name == "dota_balanced":
        return prior_balanced(
            DOTA(
                DOTAConfig(
                    batch_size=test_batch_size,
                    base_weight=0.55,
                    momentum=0.99,
                    confidence_threshold=0.0,
                )
            ),
            "dota_balanced",
            strength=0.8,
        )
    if name == "dota_linear_ensemble_balanced":
        return prior_balanced(
            linear_ensemble(
                DOTA(
                    DOTAConfig(
                        batch_size=test_batch_size,
                        base_weight=0.55,
                        momentum=0.99,
                        confidence_threshold=0.0,
                    )
                ),
                "dota_linear_ensemble",
                linear_weight=0.4,
                inner_weight=0.6,
            ),
            "dota_linear_ensemble_balanced",
            strength=0.7,
        )
    if name == "prototype_linear_tta_balanced":
        return prior_balanced(
            PrototypeLinearTTA(
                PrototypeLinearTTAConfig(
                    batch_size=test_batch_size,
                    lr=5e-6,
                    anchor_weight=0.2,
                    reg_weight=0.2,
                    balance_weight=0.1,
                    entropy_weight=0.0,
                    target_prior=(0.5, 0.5),
                )
            ),
            "prototype_linear_tta_balanced",
            strength=0.6,
        )
    if name == "online_cache_10_balanced":
        cache_path = None
        if method_cache_dir:
            cache_path = f"{method_cache_dir}/online_cache_10_balanced_source.pt"
        return prior_balanced(
            OnlineConfidentCacheAdapter(
                OnlineConfidentCacheAdapterConfig(
                    beta=beta,
                    batch_size=test_batch_size,
                    source_cache_ratio=0.1,
                    alpha_source=0.25,
                    alpha_dynamic=0.2,
                    add_threshold=0.92,
                    max_entropy=0.3,
                    entropy_score_weight=0.25,
                    use_threshold=0.88,
                    agreement_only=True,
                    max_dynamic_per_class=1024,
                    cache_path=cache_path,
                    save_cache=True,
                )
            ),
            "online_cache_10_balanced",
            strength=0.6,
        )
    raise ValueError(f"Unknown TTA method: {name}")
