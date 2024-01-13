from torchtyping import TensorType as TorchTensor
from protocols import SingleOutput

def baby_nerf(user_pose: TorchTensor[4, 4]) -> dict[str, SingleOutput]:
    # Baby-proofing
    return {
        "0": SingleOutput("dark brown wooden table", (3, 0, 0)),
        "1": SingleOutput("stainless steel oven", (2, 0, 0)),
        "2": SingleOutput("fuzzy costco bear", (1, 0, 0)),
        "3": SingleOutput("outlet", (1, 1, 0)),
        "4": SingleOutput("leather sofa", (1, 2, 0)),
    }

def rental_nerf(user_pose: TorchTensor[4, 4]) -> dict[str, SingleOutput]:
    # Rental inspection
    return {
        "0": SingleOutput("closet door with big scratch", (3, 0, 0)),
        "1": SingleOutput("white paint marks on wooden floor", (2, 0, 0)),
        "2": SingleOutput("window", (1, 0, 0)),
        "3": SingleOutput("hallway", (1, 1, 0)),
        "4": SingleOutput("quartz countertop", (1, 2, 0)),
    }

def home_nerf(user_pose: TorchTensor[4, 4]) -> dict[str, SingleOutput]:
    return {
        "0": SingleOutput("hallway", (3, 0, 0)),
        "1": SingleOutput("wooden floor", (2, 0, 0)),
        "2": SingleOutput("white wall", (1, 0, 0)),
        "3": SingleOutput("floor-to-ceiling windows", (1, 1, 0)),
        "4": SingleOutput("gas stove", (1, 2, 0)),
        "5": SingleOutput("fridge", (1, 3, 0)),
        "6": SingleOutput("table", (1, 4, 0)),
    }
