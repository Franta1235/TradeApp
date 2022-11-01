from abc import abstractmethod


class AReplicatedPortfolio:

    @abstractmethod
    def uY(self, price):
        pass

    @abstractmethod
    def uX(self, price):
        pass

    @abstractmethod
    def uYx(self, price):
        pass

    @abstractmethod
    def uXx(self, price):
        pass

    @abstractmethod
    def plot(self) -> None:
        pass

