from dataclasses import dataclass

@dataclass
class RunParams:
    ratio: float
    merge_cap: float
    k: int
    opacity_threshold: float

@dataclass
class CostParams:
    lam_geo: float = 1.0
    lam_sh: float = 1.0