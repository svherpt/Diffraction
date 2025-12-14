from abc import ABC, abstractmethod
import cupy as cp

class InitialFieldAugmenters(ABC):
    @abstractmethod
    def get_dampening_field(self, scene):
        pass

class DampeningWall(InitialFieldAugmenters):
    def __init__(self, position, size, min_dampening=0.9, alpha=3.0):
        self.position = position
        self.size = size
        self.min_dampening = min_dampening
        self.alpha = alpha

    def get_dampening_field(self, scene):
        x_start, z_start = self.position
        width, height = self.size
        x_end = x_start + width
        z_end = z_start + height

        damp_field = cp.ones(scene.electric.shape, dtype=cp.float32)

        rows = width
        cols = height

        x_lin = cp.linspace(0, 1, rows, dtype=cp.float32)
        z_lin = cp.linspace(0, 1, cols, dtype=cp.float32)

        # gradient from edges along x and z
        gx = self.min_dampening + (1.0 - self.min_dampening) * cp.exp(-self.alpha * cp.minimum(x_lin, 1 - x_lin))
        gz = self.min_dampening + (1.0 - self.min_dampening) * cp.exp(-self.alpha * cp.minimum(z_lin, 1 - z_lin))

        gradient_2d = cp.outer(gx, gz)

        damp_field[x_start:x_end, z_start:z_end] *= gradient_2d

        return [damp_field]

class DampeningBorder(InitialFieldAugmenters):
    def __init__(self, border_width, min_dampening=0.9, alpha=3.0):
        self.border_width = border_width
        self.min_dampening = min_dampening
        self.alpha = alpha

    def get_dampening_field(self, scene):
        nz, nx = scene.electric.shape
        walls = []

        top = DampeningWall(position=(0, 0), size=(self.border_width, nx),
                            min_dampening=self.min_dampening, alpha=self.alpha)
        walls.append(top.get_dampening_field(scene)[0])

        bottom = DampeningWall(position=(nz - self.border_width, 0), size=(self.border_width, nx),
                               min_dampening=self.min_dampening, alpha=self.alpha)
        walls.append(bottom.get_dampening_field(scene)[0])

        left = DampeningWall(position=(0, 0), size=(nz, self.border_width),
                             min_dampening=self.min_dampening, alpha=self.alpha)
        walls.append(left.get_dampening_field(scene)[0])

        right = DampeningWall(position=(0, nx - self.border_width), size=(nz, self.border_width),
                              min_dampening=self.min_dampening, alpha=self.alpha)
        walls.append(right.get_dampening_field(scene)[0])

        return walls
