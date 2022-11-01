from abc import ABC, abstractmethod


class AVolatility(ABC):

    @abstractmethod
    def estimate(self, _data) -> float:
        pass
