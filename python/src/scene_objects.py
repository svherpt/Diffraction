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


class LineSource(SceneObject):
    def __init__(self, start_position, size, frequency, amplitude):
        self.start_position = start_position
        self.size = size
        self.frequency = frequency
        self.amplitude = amplitude

    def render(self):
        pass  # Implementation for rendering the line source

    def update_electric_field(self, field, time_step):
        x_start, z_start = self.start_position
        width, height = self.size
        x_end = x_start + width
        z_end = z_start + height

        #Overwrite all points within the rectangle defined by start and end positions
        field[z_start:z_end, x_start:x_end] = self.amplitude * np.sin(2 * np.pi * self.frequency * time_step)