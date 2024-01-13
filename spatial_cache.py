from kd_tree import KDTree

from protocols import LLMModule, SpatialCacheProtocol, TLLMOut
from torchtyping import TensorType as TorchTensor



class SpatialCache(SpatialCacheProtocol[TLLMOut]):
    def __init__(self, llm: LLMModule, *, distance_threshold: float = 2):
        # self.cache: List[Tuple[Point, TLLMOut]] = []
        # self.nerf = nerf
        self.llm = llm
        self.distance_threshold = distance_threshold
        self.kd_tree: dict[str, KDTree] = {}
        self.inv_flat: dict[tuple[str, tuple[float, ...]], TLLMOut] = {}

    def cached_query(self, pose: TorchTensor[4, 4], task: str) -> TLLMOut:
        flat_pose = pose[:3, 1:].flatten() # 9 dimensional embedding of the pose
        tree = self.kd_tree.setdefault(task, KDTree([], 9))
        nearest = tree.get_nearest(flat_pose, return_dist_sq=True)
        if nearest is None or nearest[0] > self.distance_threshold:
            llm_out = self.llm.query(pose)
            tree.add_point(flat_pose)
            self.inv_flat[task, flat_pose] = llm_out
            return llm_out
        else:
            return self.inv_flat[task, nearest[1]]
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
