"""Dataset loaders for the four benchmarks used in the paper."""
from .adult import load_adult
from .credit import load_credit
from .health import load_health
from .cifar_hist import load_cifar_hist

__all__ = ["load_adult", "load_credit", "load_health", "load_cifar_hist"]
