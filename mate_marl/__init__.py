"""mate_marl: Heterogeneous, energy-constrained, fog-perturbed MARL on top of MATE.

This package contains the research contributions:
  - mate_marl.wrappers: env modifications (heterogeneous cameras, energy, fog,
    dict-observation packing).
  - mate_marl.nets: Set-Transformer + Type-Conditioned MoE encoder.
  - mate_marl.trainers: MAPPO with parameter sharing + centralized critic.
  - mate_marl.scripts: smoke test and training entrypoints.
"""

from mate_marl.wrappers.heterogeneous_cameras import HeterogeneousCameras
from mate_marl.wrappers.energy_constraint import EnergyConstraint
from mate_marl.wrappers.dynamic_fog import DynamicFog
from mate_marl.wrappers.dict_obs import MateMARLDictObs
from mate_marl.wrappers.flatten_for_ppo import FlattenAgentsForPPO

__all__ = [
    "HeterogeneousCameras",
    "EnergyConstraint",
    "DynamicFog",
    "MateMARLDictObs",
    "FlattenAgentsForPPO",
]
