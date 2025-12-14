from abc import ABC, abstractmethod
import numpy as np

class SceneObject(ABC):
    @abstractmethod
    def render():
        pass

    @abstractmethod
    def update_electric_field():
        pass


class PointSource(SceneObject):
    def __init__(self, position, frequency, amplitude):
        self.position = position
        self.frequency = frequency
        self.amplitude = amplitude

    def render(self):
        pass  # Implementation for rendering the point source

    def update_electric_field(self, field, time_step):
        x, z = self.position
        field[z, x] = self.amplitude * np.sin(2 * np.pi * self.frequency * time_step)