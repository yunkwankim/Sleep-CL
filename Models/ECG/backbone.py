"""
clops_backbone_model.py
-----------------------
CLOPS CNN backbone adapted for TSCIL BaseLearner interface.

The original `cnn_network_time` (prepare_network.py) uses a fixed multi-head or
fixed single-head architecture.  TSCIL agents require:
  - model.feature(x)         → feature vector (N, 100)
  - model.update_head(n_new) → dynamically grow the classifier head
  - model.head.out_features  → current number of output nodes

This wrapper reproduces the CLOPS CNN layers (3 × Conv-BN-ReLU-MaxPool + FC)
and adds a **dynamic** Linear head.  An AdaptiveAvgPool1d(10) is inserted before
the flattening step so the model works with any ECG segment length (original
CLOPS assumed exactly 10 time-steps after all convolutions/pooling).

Input  : (N, 1, T)  – batch × 1 channel × time steps
Output : (N, n_classes)
"""

import copy
import torch
import torch.nn as nn


class CLOPSBackboneModel(nn.Module):
    """
    CNN backbone matching CLOPS's cnn_network_time architecture, with a
    dynamic single-head compatible with TSCIL's BaseLearner.

    Args:
        n_initial_classes (int):   Number of output nodes in the initial head.
        dropout_type      (str):   'drop1d' (Dropout) or 'drop2d' (Dropout2d).
        p1, p2, p3        (float): Dropout probabilities for each conv block.
    """

    def __init__(self, n_initial_classes: int = 2,
                 n_channels: int = 1,
                 dropout_type: str = 'drop1d',
                 p1: float = 0.0, p2: float = 0.0, p3: float = 0.0):
        super(CLOPSBackboneModel, self).__init__()

        c1, c2, c3, c4 = n_channels, 4, 16, 32
        k, s = 7, 3                        # kernel size, stride (same as original)

        # ── Convolutional backbone (mirrors cnn_network_time) ──────────────
        self.conv1      = nn.Conv1d(c1, c2, k, s)
        self.batchnorm1 = nn.BatchNorm1d(c2)

        self.conv2      = nn.Conv1d(c2, c3, k, s)
        self.batchnorm2 = nn.BatchNorm1d(c3)

        self.conv3      = nn.Conv1d(c3, c4, k, s)
        self.batchnorm3 = nn.BatchNorm1d(c4)

        # AdaptiveAvgPool ensures the spatial dim is always 10 regardless of
        # input length → c4 * 10 = 320-dim flattened representation.
        self.adaptive_pool = nn.AdaptiveAvgPool1d(10)
        self.linear1       = nn.Linear(c4 * 10, 100)   # 320 → 100

        # ── Activations / pooling ──────────────────────────────────────────
        self.relu    = nn.ReLU()
        self.maxpool = nn.MaxPool1d(2)

        # ── Dropout (per-block) ────────────────────────────────────────────
        if dropout_type == 'drop2d':
            self.dropout1 = nn.Dropout2d(p=p1)
            self.dropout2 = nn.Dropout2d(p=p2)
            self.dropout3 = nn.Dropout2d(p=p3)
        else:                                       # default: drop1d
            self.dropout1 = nn.Dropout(p=p1)
            self.dropout2 = nn.Dropout(p=p2)
            self.dropout3 = nn.Dropout(p=p3)

        # ── Dynamic classifier head ────────────────────────────────────────
        self.head = nn.Linear(100, n_initial_classes)

    # ------------------------------------------------------------------
    # Feature extraction (before classifier head)
    # ------------------------------------------------------------------
    def feature(self, x: torch.Tensor) -> torch.Tensor:
        """Return 100-dim feature vector (before the classifier head).

        Args:
            x: (N, 1, T) – ECG input, channels-first.
        Returns:
            (N, 100) feature tensor.
        """
        x = self.dropout1(self.maxpool(self.relu(self.batchnorm1(self.conv1(x)))))
        x = self.dropout2(self.maxpool(self.relu(self.batchnorm2(self.conv2(x)))))
        x = self.dropout3(self.maxpool(self.relu(self.batchnorm3(self.conv3(x)))))
        x = self.adaptive_pool(x)                                     # (N, 32, 10)
        x = torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))  # (N, 320)
        x = self.relu(self.linear1(x))                                 # (N, 100)
        return x

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.feature(x))

    # ------------------------------------------------------------------
    # Dynamic head expansion (called by TSCIL BaseLearner.before_task)
    # ------------------------------------------------------------------
    def update_head(self, n_new: int, task_now: int = None):
        """Expand the classifier head by *n_new* output nodes.

        Old weights are preserved; new weights are Xavier-initialised.

        Args:
            n_new    : Number of new output nodes to add.
            task_now : Current task index (unused, kept for API compatibility).
        """
        n_old    = self.head.out_features
        old_head = copy.deepcopy(self.head)
        new_head = nn.Linear(old_head.in_features, n_old + n_new)
        nn.init.xavier_uniform_(new_head.weight)
        with torch.no_grad():
            new_head.weight.data[:n_old] = old_head.weight.data
            new_head.bias.data[:n_old]   = old_head.bias.data
        self.head = new_head

    # ------------------------------------------------------------------
    # Convenience alias (used by some TSCIL agents, e.g. DT2W)
    # ------------------------------------------------------------------
    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Return conv feature map (N, 32, 10) before global pooling."""
        x = self.dropout1(self.maxpool(self.relu(self.batchnorm1(self.conv1(x)))))
        x = self.dropout2(self.maxpool(self.relu(self.batchnorm2(self.conv2(x)))))
        x = self.dropout3(self.maxpool(self.relu(self.batchnorm3(self.conv3(x)))))
        return self.adaptive_pool(x)                                   # (N, 32, 10)
