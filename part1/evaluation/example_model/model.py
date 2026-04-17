"""
Stub / persistence baseline model for Assignment 3.

Prediction strategy: repeat the last 24 hours of known energy demand as the
forecast for the next 24 hours.  No learning involved — this serves as a
lower-bound baseline.

Interface (shared by all models in this evaluation framework):
    get_model(metadata: dict) -> torch.nn.Module

    Every model must implement two methods:

    adapt_inputs(history_weather, history_energy, future_weather, future_time) -> tuple
        Converts the raw, full-resolution evaluation inputs into whatever
        (lighter) inputs your forward() actually expects.  The evaluator calls
        this first and unpacks the returned tuple into forward().

        Raw inputs passed by the evaluator:
            history_weather : (B, 168, 450, 449, 7) float32
            history_energy  : (B, 168, n_zones) float32
            future_weather  : (B, 24, 450, 449, 7) float32
            future_time     : (B, 24) int64  -- hours since Unix epoch

        This is also where you should replicate any feature extraction that
        your data loader performs during training (e.g. time features, spatial
        pooling), so that evaluation is consistent with training.

        The default implementation is an identity pass-through.

    forward(*adapt_inputs(...)) -> (B, 24, n_zones) float32
        Runs the model on the outputs of adapt_inputs().  During training you
        can call forward() directly with lightweight, batched inputs; the
        evaluator always goes through adapt_inputs() first.
"""

import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import submodule  # To test whether submodules can be loaded


class StubModel(nn.Module):
    """
    Persistence baseline: predict the next 24 hours of energy demand by
    repeating the last 24 hours of the history window.

    Parameters
    ----------
    n_zones : int
        Number of energy demand zones.
    """

    def __init__(self, n_zones: int):
        super().__init__()
        self.n_zones = n_zones

    def adapt_inputs(
        self,
        history_weather: torch.Tensor,
        history_energy: torch.Tensor,
        future_weather: torch.Tensor,
        future_time: torch.Tensor,
    ) -> tuple:
        """
        Transform raw evaluation inputs before they are passed to forward().

        Override this method in subclasses to reduce or reshape the inputs
        (e.g. spatial pooling of weather grids, history subsampling, feature
        extraction) without changing the evaluation harness.

        Parameters
        ----------
        history_weather : (B, 168, 450, 449, 7)
        history_energy  : (B, 168, n_zones)
        future_weather  : (B, 24, 450, 449, 7)
        future_time     : (B, 24) int64

        Returns
        -------
        tuple
            A tuple whose elements are forwarded to forward() via unpacking.
            The default implementation is an identity pass-through.
        """
        # NOTE For the class, we need to have a unified interface, so we require 
        # you to provide this function that convert the standard but redundant 
        # input values into the input your forward function accpets. 


        # In this function should be consistent with the data loader you use. For example, 
        # if your data loader extract time features, then you should call the same 
        # feature extraction method here. 

        time_feature = self.extract_time_feature(future_time) 


        # During model training, you can use the forward function only: there you can 
        # use light-weight inputs with large batch sizes.  


        return (history_weather, history_energy, future_weather, future_time)

    def forward(
        self,
        history_weather: torch.Tensor,
        history_energy: torch.Tensor,
        future_weather: torch.Tensor,
        future_time: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        history_weather : (B, 168, 450, 449, 7)  -- unused by this baseline
        history_energy  : (B, 168, n_zones) float32
        future_weather  : (B, 24, 450, 449, 7)   -- unused by this baseline
        future_time     : (B, 24) int64  -- hours since Unix epoch  -- unused by this baseline

        Returns
        -------
        torch.Tensor
            Shape (B, 24, n_zones) — predicted energy demand.
        """


        # Use the last 24 hours of history as the prediction
        return history_energy[:, -24:, :]  # (B, 24, n_zones)

    def extract_time_feature(self, future_time: torch.Tensor) -> torch.Tensor:
        """
        An example of extracting time features from hours-since-epoch and returning a one-hot
        encoding of the day of week for each of the 24 future hours.

        Parameters
        ----------
        future_time : (B, 24) int64 -- hours since Unix epoch

        Returns
        -------
        torch.Tensor
            Shape (B, 24, 7) float32 -- one-hot day-of-week per hour.
            Axis -1: [Mon, Tue, Wed, Thu, Fri, Sat, Sun] (Monday=0, Sunday=6).
        """
        # NOTE: Using numpy/pandas for calendar arithmetic is fine here —
        # no gradient flows through time features.
        # NOTE: an faster alternative is to compute entire feature matrix and store 
        # it in the model and do look up here.  

        B, T = future_time.shape
        hours_np = future_time.cpu().numpy()  # (B, 24)

        dow = np.empty((B, T), dtype=np.int64)
        for b in range(B):
            dti = pd.DatetimeIndex(hours_np[b].astype("datetime64[h]"))
            dow[b] = dti.dayofweek  # Monday=0, Sunday=6

        # One-hot encode: (B, 24) int -> (B, 24, 7) float
        one_hot = np.zeros((B, T, 7), dtype=np.float32)
        one_hot[np.arange(B)[:, None], np.arange(T)[None, :], dow] = 1.0

        return torch.from_numpy(one_hot)  # (B, 24, 7)


def get_model(metadata: dict) -> StubModel:
    """
    Build and return a StubModel from dataset metadata.

    Parameters
    ----------
    metadata : dict
        Loaded from evaluate.py.  Must contain:
            - "n_zones": int
            - "zone_names": list[str]
            - "history_len": int  (168)
            - "future_len": int   (24)
            - "n_weather_vars": int (7)
    """
    return StubModel(n_zones=metadata["n_zones"])
