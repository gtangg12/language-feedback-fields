from dataclasses import dataclass
from typing import Generic, List, Protocol, Tuple, TypeVar


@dataclass
class Point:
    x: float
    y: float
    z: float
    def distance(self, other: "Point") -> float:
        return ((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)**0.5

class NeRFModule(Protocol):
    def query_point(self, point: Point) -> List[Tuple[Point, str]]:
        ...


class LLMOutput(Protocol):
    # can add this in later
    # @staticmethod
    # def spatial_interpolate(cached_outputs: "tuple[Point, LLMOutput]") -> "LLMOutput":
    #     pass
    pass

TLLMOut = TypeVar("TLLMOut", bound=LLMOutput, covariant=True)


class LLMModule(Protocol, Generic[TLLMOut]):
    def query(self, data: List[Tuple[Point, str]]) -> TLLMOut:
        ...


class SpatialCacheProtocol(Protocol, Generic[TLLMOut]):
    nerf: NeRFModule
    llm: LLMModule
    def cached_query(self, point: Point) -> TLLMOut:
        ...