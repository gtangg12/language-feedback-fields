from typing import List, Tuple

from protocols import LLMModule, NeRFModule, Point, SpatialCacheProtocol, TLLMOut


class SpatialCache(SpatialCacheProtocol[TLLMOut]):
    def __init__(self, nerf: NeRFModule, llm: LLMModule, *, distance_threshold: float = 0.1):
        # very stupid, maybe later I can make an actually good kNN data structure
        self.cache: List[Tuple[Point, TLLMOut]] = []
        self.nerf = nerf
        self.llm = llm
        self.distance_threshold = distance_threshold

    def cached_query(self, point: Point) -> TLLMOut:
        nearest: tuple[Point, TLLMOut] = self.nearest_neighbor(point)
        if nearest[0].distance(point) < self.distance_threshold:
            return nearest[1]
        else:
            nerf_out = self.nerf.query_point(point)
            llm_out = self.llm.query(nerf_out)
            self.cache.append((point, llm_out))
            return llm_out

    def nearest_neighbor(self, point: Point) -> tuple[Point, TLLMOut]:
        best = self.cache[0]
        for cache_value in self.cache[1:]:
            if cache_value[0].distance(point) < best[0].distance(point):
                best = cache_value
        return best
