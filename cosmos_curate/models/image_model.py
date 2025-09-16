"""Image model."""

import numpy as np
import numpy.typing as npt
import torch

from cosmos_curate.core.interfaces.model_interface import ModelInterface
from cosmos_curate.core.utils.model import conda_utils, model_utils

_IMAGE_MODEL_ID = "nvidia/mit-b0"

# IF your model needs to run in a specific conda env
# import env-specific dependencies here
if conda_utils.is_running_in_env("dedicated-conda-env"):
    from transformers import AutoModel


class ImageInferenceModel(ModelInterface):
    def __init__(self) -> None:
        super().__init__()
        self._model: AutoModel | None = None

    @property
    def conda_env_name(self) -> str:
        # If your model needs to run in a specific conda env
        # otherwise, return `default`
        return "dedicated-conda-env"

    @property
    def model_id_names(self) -> list[str]:
        # framework will ensure model file exists & download if needed
        return [_IMAGE_MODEL_ID]

    def setup(self) -> None:
        # this runs on the actor in specified conda env
        model_dir = model_utils.get_local_dir_for_weights_name(self.model_id_names[0])
        self._model = load_model_from_checkpoint((model_dir / "pytorch_model.bin").as_posix())

    def generate(self, batch: torch.Tensor) -> npt.NDArray[np.float32]:
        labels = self._model(batch.to("cuda"))
        return labels.cpu().numpy()
