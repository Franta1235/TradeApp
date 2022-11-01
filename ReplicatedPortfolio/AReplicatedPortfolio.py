import pandas as pd
import datetime
from Helper import roundTimeUp, roundTimeDown, daterange
from abc import abstractmethod
from Window.AWindow import AWindow


class AReplicatedPortfolio:

    def __init__(self):
        pass

    @abstractmethod
    def create_window(self, prices) -> AWindow:
        pass

    @abstractmethod
    def update(self, prices) -> None:
        pass

    @abstractmethod
    def uY(self, X) -> pd.DataFrame:
        pass

    @abstractmethod
    def uX(self, X) -> pd.DataFrame:
        pass

    @abstractmethod
    def uYx(self, X) -> pd.DataFrame:
        pass

    @abstractmethod
    def uXx(self, X) -> pd.DataFrame:
        pass

    @abstractmethod
    def plot(self) -> None:
        pass

    @abstractmethod
    def trade_theoretical(self) -> float:
        pass

    @abstractmethod
    def trade(self) -> float:
        pass
