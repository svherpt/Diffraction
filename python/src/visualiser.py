import numpy as np
import cv2

class Visualiser:
    def __init__(self, simulator, window_name='Wave Propagation'):
        self.simulator = simulator
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, simulator.nx, simulator.nz)

    def update(self, render_mode='field'):

        if render_mode == 'dampening':
            frame = ((1-self.simulator.dampening_field) * 255).astype(np.uint8)
        elif render_mode == 'electric':
            frame = self.simulator.get_electric_field()
        
        cv2.imshow(self.window_name, frame.get())

    def close(self):
        cv2.destroyAllWindows()