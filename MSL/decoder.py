from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as _config

config = _config.get_config("pi05_xarm")
checkpoint_dir = download.maybe_download("/home/larsosterberg/MSL/openpi/checkpoints/pi05_xarm_finetune/lars_test/2999")

# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir)
"""
# This is the important part of inference

def sample_actions(self, rng: at.KeyArrayLike, observation: Observation, **kwargs) -> Actions: ...

from openpi.models import model as _model

def __init__(
        self,
        model: _model.BaseModel,

self._sample_actions = nnx_utils.module_jit(model.sample_actions)

outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
"""

action = np.array(policy.infer(observation)["actions"])