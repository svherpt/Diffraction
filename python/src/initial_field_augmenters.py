from abc import ABC, abstractmethod
import cupy as cp

class InitialFieldAugmenters(ABC):
    @abstractmethod
    def get_dampening_field(self, scene):
        pass

class DampeningWall(InitialFieldAugmenters):
    def __init__(self, position, size, min_dampening=0.9, alpha=3.0, gradient_axis='z', reverse=False):
        self.position = position
        self.size = size
        self.min_dampening = min_dampening
        self.alpha = alpha
        self.gradient_axis = gradient_axis
        self.reverse = reverse

    def get_dampening_field(self, scene):
        x_start, z_start = self.position
        width, height = self.size
        x_end = x_start + width
        z_end = z_start + height

        damp_field = cp.ones(scene.electric.shape, dtype=cp.float32)

        rows = x_end - x_start
        cols = z_end - z_start

        if self.gradient_axis == 'z':
            lin = cp.linspace(0, 1, cols, dtype=cp.float32)
            if self.reverse:
                lin = lin[::-1]
            gradient = self.min_dampening + (1.0 - self.min_dampening) * cp.exp(-self.alpha * (1 - lin))
            gradient_2d = cp.tile(gradient[None, :], (rows, 1))
        elif self.gradient_axis == 'x':
            lin = cp.linspace(0, 1, rows, dtype=cp.float32)
            if self.reverse:
                lin = lin[::-1]
            gradient = self.min_dampening + (1.0 - self.min_dampening) * cp.exp(-self.alpha * (1 - lin))
            gradient_2d = cp.tile(gradient[:, None], (1, cols))
        else:
            raise ValueError("gradient_axis must be 'x' or 'z'")

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
                            min_dampening=self.min_dampening, alpha=self.alpha, gradient_axis='x', reverse=False)
        walls.append(top.get_dampening_field(scene)[0])

        bottom = DampeningWall(position=(nz - self.border_width, 0), size=(self.border_width, nx),
                               min_dampening=self.min_dampening, alpha=self.alpha, gradient_axis='x', reverse=True)
        walls.append(bottom.get_dampening_field(scene)[0])

        left = DampeningWall(position=(0, 0), size=(nz, self.border_width),
                             min_dampening=self.min_dampening, alpha=self.alpha, gradient_axis='z', reverse=False)
        walls.append(left.get_dampening_field(scene)[0])

        right = DampeningWall(position=(0, nx - self.border_width), size=(nz, self.border_width),
                              min_dampening=self.min_dampening, alpha=self.alpha, gradient_axis='z', reverse=True)
        walls.append(right.get_dampening_field(scene)[0])

        return walls
