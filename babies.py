from torchtyping import TensorType as TorchTensor
from protocols import SingleOutput
import torch

babies_task = "Are there objects nearby which can hurt a baby?"
def mock_nerf_babies(user_pose: TorchTensor[4, 4]) -> dict[str, SingleOutput]:
    if user_pose[0, 3] > 0:
        return {
            "knife": SingleOutput("sharp and kills you", 3),
        }
    else:
        return {}

front_pose = torch.Tensor(4, 4)
front_pose.fill_(1)

back_pose = torch.Tensor(4, 4)
back_pose.fill_(-1)