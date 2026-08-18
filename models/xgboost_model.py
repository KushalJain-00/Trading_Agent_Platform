"""XGBoost tabular classifier — window featurized to summary stats.

Not a neural network; lives here only for checkpoint path convention.
Training logic is in train_xgboost.py (separate from train.py).
"""
import os

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "xgboost.json")
