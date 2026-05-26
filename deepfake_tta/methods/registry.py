from __future__ import annotations

from deepfake_tta.methods.bca import BCA, BCAConfig
from deepfake_tta.methods.boost_adapter import BoostAdapter, BoostAdapterConfig
from deepfake_tta.methods.crg import CRG, CRGConfig
from deepfake_tta.methods.dmn import DMN, DMNConfig
from deepfake_tta.methods.dota import DOTA, DOTAConfig
from deepfake_tta.methods.dpe import DPE, DPEConfig
from deepfake_tta.methods.dynaprompt import DynaPrompt, DynaPromptConfig
from deepfake_tta.methods.freetta import FreeTTA, FreeTTAConfig
from deepfake_tta.methods.tip_adapter import TipAdapter, TipAdapterConfig

AVAILABLE_TTA_METHODS = (
    "tip_adapter",
    "boost_adapter",
    "crg",
    "dmn",
    "dpe",
    "dota",
    "freetta",
    "bca",
    "dynaprompt",
)


def create_tta_method(
    name: str,
    *,
    alpha: float = 0.5,
    beta: float = 5.5,
    test_batch_size: int = 512,
    cache_batch_size: int = 8192,
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
    if name == "dynaprompt":
        return DynaPrompt(DynaPromptConfig(batch_size=test_batch_size))
    raise ValueError(f"Unknown TTA method: {name}")
