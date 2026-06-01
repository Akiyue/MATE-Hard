from collections import OrderedDict
from typing import Callable

import torch.nn as nn
from torch import Tensor

from .activations import (
    AccurateGELUActivation,
    ClippedGELUActivation,
    FastGELUActivation,
    GELUActivation,
    LaplaceActivation,
    LinearActivation,
    MishActivation,
    NewGELUActivation,
    PytorchGELUTanh,
    QuickGELUActivation,
    ReLUSquaredActivation,
)

ACT2CLS = {
    "gelu": GELUActivation,
    "gelu_10": (ClippedGELUActivation, {"min": -10, "max": 10}),
    "gelu_fast": FastGELUActivation,
    "gelu_new": NewGELUActivation,
    "gelu_python": (GELUActivation, {"use_gelu_python": True}),
    "gelu_pytorch_tanh": PytorchGELUTanh,
    "gelu_accurate": AccurateGELUActivation,
    "laplace": LaplaceActivation,
    "leaky_relu": nn.LeakyReLU,
    "linear": LinearActivation,
    "mish": MishActivation,
    "quick_gelu": QuickGELUActivation,
    "relu": nn.ReLU,
    "relu2": ReLUSquaredActivation,
    "relu6": nn.ReLU6,
    "sigmoid": nn.Sigmoid,
    "silu": nn.SiLU,
    "swish": nn.SiLU,
    "tanh": nn.Tanh,
    "prelu": nn.PReLU,
    "softmax": nn.Softmax,
}

OP2CLS = {
    "+": lambda: lambda x, y: x + y,
    "-": lambda: lambda x, y: x - y,
    "*": lambda: lambda x, y: x * y,
    "/": lambda: lambda x, y: x / y,
    "@": lambda: lambda x, y: x @ y,
}


class ClassInstantier(OrderedDict):
    def __getitem__(self, key: str):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)


ACT2FN = ClassInstantier(ACT2CLS)
OP2FN = ClassInstantier(OP2CLS)


def get_activation(activation_string: str | Callable[[Tensor], Tensor]):
    if isinstance(activation_string, Callable):
        return activation_string
    elif isinstance(activation_string, str):
        activation_string = activation_string.lower().strip()
        if activation_string in ACT2FN:
            return ACT2FN[activation_string]
        raise KeyError(
            f"function {activation_string} not found in ACT2FN mapping {list(ACT2FN.keys())}"
        )
    raise ValueError(
        f"activation_string should be either a string or Callable instance, but got {type(activation_string)}"
    )


def get_operator_function(operator_string: str | Callable[[Tensor, Tensor], Tensor]):
    if isinstance(operator_string, Callable):
        return operator_string
    elif isinstance(operator_string, str):
        operator_string = operator_string.lower().strip()
        if operator_string in OP2FN:
            return OP2FN[operator_string]
        raise KeyError(
            f"function {operator_string} not found in OP2FN mapping {list(OP2FN.keys())}"
        )
    raise ValueError(
        f"operator_string should be either a string or Callable instance, but got {type(operator_string)}"
    )
