from dataclasses import dataclass
from simple_parsing import Serializable

__all__ = []
__all__.extend([
    'PlotParams',
    'HyperParams'
])


@dataclass
class PlotParams(Serializable):
    val: int


@dataclass
class HyperParams(Serializable):
    plot: PlotParams
