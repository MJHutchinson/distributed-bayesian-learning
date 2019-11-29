from abc import ABC, abstractmethod


class MomentsParams(ABC):
    @abstractmethod
    def to_mean_params(self):
        raise NotImplementedError

    @abstractmethod
    def to_nat_params(self):
        raise NotImplementedError

    @abstractmethod
    def to_moment_params(self):
        raise NotImplementedError

    @classmethod
    def from_mean_params(mean_params):
        raise NotImplementedError

    @classmethod
    def from_nat_params(nat_params):
        raise NotImplementedError

    @classmethod
    def from_moment_params(moment_params):
        raise NotImplementedError

    @abstractmethod
    def to_numpy(self):
        raise NotImplementedError

    @abstractmethod
    def to_torch(self, device='cpu'):
        raise NotImplementedError


class MeanParams(ABC):
    @abstractmethod
    def to_mean_params(self):
        raise NotImplementedError

    @abstractmethod
    def to_nat_params(self):
        raise NotImplementedError

    @abstractmethod
    def to_moment_params(self):
        raise NotImplementedError

    @classmethod
    def from_mean_params(mean_params):
        raise NotImplementedError

    @classmethod
    def from_nat_params(nat_params):
        raise NotImplementedError

    @classmethod
    def from_moment_params(moment_params):
        raise NotImplementedError

    @abstractmethod
    def to_numpy(self):
        raise NotImplementedError

    @abstractmethod
    def to_torch(self, device='cpu'):
        raise NotImplementedError


class NatParams(ABC):
    @abstractmethod
    def to_mean_params(self):
        raise NotImplementedError

    @abstractmethod
    def to_nat_params(self):
        raise NotImplementedError

    @abstractmethod
    def to_moment_params(self):
        raise NotImplementedError

    @classmethod
    def from_mean_params(mean_params):
        raise NotImplementedError

    @classmethod
    def from_nat_params(nat_params):
        raise NotImplementedError

    @classmethod
    def from_moment_params(moment_params):
        raise NotImplementedError

    @abstractmethod
    def to_numpy(self):
        raise NotImplementedError

    @abstractmethod
    def to_torch(self, device='cpu'):
        raise NotImplementedError