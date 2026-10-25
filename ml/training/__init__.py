from .trainer import Trainer
from .online_learner import OnlineLearner
from .loss_functions import MultiLabelFocalLoss, HierarchicalLoss
# Pandas/parquet are optional for the synthetic PoC path.
def __getattr__(name):
    if name == "InterestDataModule":
        from .data_module import InterestDataModule
        return InterestDataModule
    raise AttributeError(name)

__all__ = [
    "Trainer",
    "OnlineLearner",
    "MultiLabelFocalLoss",
    "HierarchicalLoss",
    "InterestDataModule",
]
