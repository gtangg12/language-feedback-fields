from dataclasses import dataclass
import math
from typing import Generic, List, Protocol, Tuple, TypeVar


@dataclass
class Point:
    x: float
    y: float
    z: float
    theta: float
    phi: float

    def distance(self, other: "Point") -> float:
        translation =  (
            (self.x - other.x)**2 + 
            (self.y - other.y)**2 + 
            (self.z - other.z)**2
        ) ** 0.5
        # cos(theta), sin(theta)cos(phi), sin(theta)sin(phi)
        rotation = (
            1 - (math.sin(self.theta) * math.sin(other.theta) * math.cos(self.phi - other.phi) + math.cos(self.theta) * math.cos(other.theta))
        ) ** 0.5
        return translation + rotation
    
    def embed6d(self) -> tuple[float, float, float, float, float, float]:
        return (self.x, self.y, self.z, math.cos(self.theta), math.sin(self.theta) * math.cos(self.phi), math.sin(self.theta) * math.sin(self.phi))
    
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
    # nerf: NeRFModule
    llm: LLMModule
    def query(self, point: Point) -> TLLMOut:
        ...


@dataclass
class SingleOutput:
    description: str
    scoord: tuple[float, float, float] # r, theta, phi