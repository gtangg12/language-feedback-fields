from typing import Generic, Protocol, TypeVar


@dataclass
class Point:
    x: float
    y: float
    z: float

class NeRFModule(Protocol):
    def query_point(self, point: Point) -> List[Tuple[Point, str]]:
        pass


class LLMOutput(Protocol):
    pass


TLLMOut = TypeVar("TLLMOut", bound=LLMOutput)


class LLMModule(Protocol, Generic[TLLMOut]):
    def query(self, data: tuple[Point, str]) -> TLLMOut:
        pass


class SpatialCache:
    nerf: NeRFModule
    llm: LLMModule
    def cached_query(self, point: Point) -> TLLMOut:
        ...