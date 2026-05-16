"""Gaussian HMM regime detector.

Trained on a low-dimensional aggregate market signal — by default a 2-vector of
[SPX daily log return, log VIX]. We use full covariance and `k=3` regimes
("calm", "normal", "stress"). Training is done on a TRAIN window only; inference
emits posterior probabilities online via the forward algorithm so there is no
look-ahead leakage.

We expose a thin wrapper around `hmmlearn` so the rest of the codebase doesn't
import it directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from ..utils.logging import get_logger

LOG = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def make_regime_signal(macro: pd.DataFrame) -> pd.DataFrame:
    """Build the low-D observation series the HMM is trained on.

    Columns:
      - spx_ret : 1-day log return of SPX
      - log_vix : log of VIX (level)
    Rows with any NaN are dropped by the caller before fit.
    """
    out = pd.DataFrame(index=macro.index)
    if "SPX" not in macro:
        raise KeyError("Macro frame missing 'SPX' for regime signal.")
    if "VIX" not in macro:
        raise KeyError("Macro frame missing 'VIX' for regime signal.")
    out["spx_ret"] = np.log(macro["SPX"]).diff()
    out["log_vix"] = np.log(macro["VIX"])
    return out


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

@dataclass
class HMMConfig:
    n_states: int = 3
    covariance_type: str = "full"
    n_iter: int = 200
    tol: float = 1e-4
    random_state: int = 42


class RegimeHMM:
    """Wrapper around hmmlearn.GaussianHMM with state-relabeling by mean vol.

    After fit we relabel hidden states so that state 0 = lowest-vol regime,
    state K-1 = highest-vol regime. This gives consistent semantics across
    re-trainings (otherwise hmmlearn's state ids are arbitrary).
    """

    def __init__(self, config: HMMConfig | None = None):
        self.config = config or HMMConfig()
        self.model: GaussianHMM | None = None
        self.state_perm: np.ndarray | None = None  # original_id -> sorted_id

    # ----- fit / predict ----- #

    def fit(self, signal: pd.DataFrame) -> "RegimeHMM":
        X = signal.dropna().values.astype(np.float64)
        if X.shape[0] < 50 * self.config.n_states:
            raise ValueError(
                f"Too few observations ({X.shape[0]}) for {self.config.n_states}-state HMM."
            )
        m = GaussianHMM(
            n_components=self.config.n_states,
            covariance_type=self.config.covariance_type,
            n_iter=self.config.n_iter,
            tol=self.config.tol,
            random_state=self.config.random_state,
        )
        m.fit(X)
        # Relabel by per-state vol of the SPX return component (column 0).
        # Higher absolute mean of column 1 (log VIX) -> stress.
        # We use log VIX mean as the ordering key (more stable than spx_ret mean).
        means_vix = m.means_[:, 1]
        order = np.argsort(means_vix)              # low -> high vol
        self.state_perm = np.argsort(order)        # original_id -> new_id
        self.model = m
        LOG.info(
            "HMM fit: log-likelihood=%.1f, state log-VIX means (sorted)=%s",
            m.score(X), means_vix[order].tolist(),
        )
        return self

    def _check_fit(self) -> None:
        if self.model is None:
            raise RuntimeError("RegimeHMM not fit yet.")

    def predict_proba(self, signal: pd.DataFrame) -> pd.DataFrame:
        """Posterior probabilities (forward-backward) of each regime per date.

        Rows that contain NaN in the input signal get all-NaN posteriors and
        are forward-filled later by the caller (or the bundle assembler).
        """
        self._check_fit()
        valid_idx = signal.dropna().index
        X = signal.loc[valid_idx].values.astype(np.float64)
        post = self.model.predict_proba(X)               # original ordering
        post = post[:, np.argsort(self.state_perm)]      # remap to sorted ids
        cols = [f"regime_{i}" for i in range(self.config.n_states)]
        df = pd.DataFrame(post, index=valid_idx, columns=cols)
        return df.reindex(signal.index)

    def predict_states(self, signal: pd.DataFrame) -> pd.Series:
        self._check_fit()
        valid_idx = signal.dropna().index
        X = signal.loc[valid_idx].values.astype(np.float64)
        states = self.model.predict(X)
        states = self.state_perm[states]
        return pd.Series(states, index=valid_idx, name="regime").reindex(signal.index)

    # ----- persistence ----- #

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"config": self.config, "model": self.model, "state_perm": self.state_perm}, path)

    @classmethod
    def load(cls, path: str | Path) -> "RegimeHMM":
        blob = joblib.load(path)
        obj = cls(config=blob["config"])
        obj.model = blob["model"]
        obj.state_perm = blob["state_perm"]
        return obj
