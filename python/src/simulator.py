import numpy as np
import cupy as cp
import cupyx.scipy.signal

class WaveSimulator:
    def __init__(self, grid_size, grid_spacing, scene_objects, initial_field_augmenters, time_scaler=0.01):
        self.SPEED_OF_LIGHT = 3e8
        self.scene_objects = scene_objects
        self.initial_field_augmenters = initial_field_augmenters

        self.nz, self.nx = grid_size
        self.dx, self.dz = grid_spacing
        self.dt = self.dx * time_scaler / (2 * np.sqrt(2) * self.SPEED_OF_LIGHT)
        self.t = 0
        
        self.electric_prev = cp.zeros((self.nz, self.nx))
        self.electric = cp.zeros((self.nz, self.nx))
        self.dampening_field = cp.ones((self.nz, self.nx))

        self.laplacian_kernel = cp.array([[0.066, 0.184, 0.066],
                                          [0.184, -1.0, 0.184],
                                          [0.066, 0.184, 0.066]])
        self.initialise_fields()

    def initialise_fields(self):
        all_dampening_fields = []

        for augmenter in self.initial_field_augmenters:
            all_dampening_fields += augmenter.get_dampening_field(self)

        for dampening_field in all_dampening_fields:
            self.dampening_field = cp.minimum(self.dampening_field, dampening_field)

    def update(self, event):

        num_steps = 1
        if(event["type"] == "skip"):
            num_steps = event["num_steps"]

        print(f"Advancing simulation by {num_steps} steps.")

        for _ in range(num_steps):
            self.step(event)

    def step(self, event):
        self.t += self.dt


        laplacian = cupyx.scipy.signal.convolve2d(self.electric, self.laplacian_kernel, mode='same', boundary='fill')

        electric_next = self.electric + self.dampening_field * (self.electric- self.electric_prev) + (self.SPEED_OF_LIGHT * self.dt / self.dx) ** 2 * laplacian
    
        self.electric_prev[:] = self.electric
        self.electric[:] = electric_next

        for obj in self.scene_objects:
            obj.update_electric_field(self.electric, self.t)

            

    def get_electric_field(self):
        return self.electric