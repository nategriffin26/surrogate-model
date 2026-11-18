"""Predict the surface pressure coefficient of a DrivAerML-style car from its geometry.

`SurrogateEnsemble.load` reads the trained networks; `run_vtp_inference` predicts every cell of a `.vtp`
surface file; `write_prediction_vtp` writes the file back with the prediction.
"""

from surrogate.inference import InferenceResult, run_vtp_inference, write_prediction_vtp
from surrogate.model import SurrogateEnsemble

__all__ = [
    "InferenceResult",
    "SurrogateEnsemble",
    "run_vtp_inference",
    "write_prediction_vtp",
]
