from babies import mock_nerf_babies, babies_task, front_pose, back_pose
from kd_tree import KDTree
from llm_agent import LLMAgent

from protocols import LLMModule, SpatialCacheProtocol, TLLMOut
from torchtyping import TensorType as TorchTensor


class SpatialCache(SpatialCacheProtocol[TLLMOut]):
    def __init__(self, llm: LLMModule, *, distance_threshold: float = 1):
        # self.cache: List[Tuple[Point, TLLMOut]] = []
        # self.nerf = nerf
        self.llm = llm
        self.distance_threshold = distance_threshold
        self.kd_tree: dict[str, KDTree] = {}
        self.inv_flat: dict[tuple[str, tuple[float, ...]], TLLMOut] = {}

    def query(self, user_pose: TorchTensor[4, 4], task_prompt: str) -> TLLMOut:
        flat_pose = tuple(user_pose[:3].flatten().tolist()) # 12 dimensional embedding of the pose
        tree = self.kd_tree.setdefault(task_prompt, KDTree([], 12))
        nearest = tree.get_nearest(flat_pose, return_dist_sq=True)
        if nearest is None or nearest[0] > self.distance_threshold:
            llm_out = self.llm.query(user_pose=user_pose, task_prompt=task_prompt)
            tree.add_point(flat_pose)
            self.inv_flat[task_prompt, flat_pose] = llm_out
            return llm_out
        else:
            return self.inv_flat[task_prompt, nearest[1]]
        # nearest: tuple[Point, TLLMOut] = self.nearest_neighbor(point)
        # if nearest[0].distance(point) < self.distance_threshold:
        #     return nearest[1]
        # else:
        #     nerf_out = self.nerf.query_point(point)
        #     llm_out = self.llm.query(nerf_out)
        #     self.cache.append((point, llm_out))
        #     return llm_out

    # def nearest_neighbor(self, point: Point) -> tuple[Point, TLLMOut]:
        # best = self.cache[0]
        # for cache_value in self.cache[1:]:
        #     if cache_value[0].distance(point) < best[0].distance(point):
        #         best = cache_value
        # return best


if __name__ == '__main__':
    agent_babies = LLMAgent(mock_nerf_babies)
    cached_agent = SpatialCache(agent_babies)
    print(cached_agent.query(user_pose=front_pose, task_prompt=babies_task))
    print(cached_agent.query(user_pose=front_pose, task_prompt=babies_task))
    print(cached_agent.query(user_pose=back_pose, task_prompt=babies_task))
    print(cached_agent.query(user_pose=back_pose, task_prompt=babies_task))
    print(cached_agent.query(user_pose=front_pose, task_prompt=babies_task))