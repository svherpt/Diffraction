from src.simulator import WaveSimulator
from src.visualiser import Visualiser
from src.scene_objects import PointSource
from src.initial_field_augmenters import DampeningWall, DampeningBorder
from src.key_handler import KeyHandler
import cv2
import cupy as cp


def main():

    nx, nz = 1080, 1920

    scene_objects = [
        PointSource(position=(nz // 4, 17 * nx // 32), frequency=2.2e14, amplitude=10),
        PointSource(position=(nz // 4, 16 * nx // 32), frequency=2.2e14, amplitude=10),
        PointSource(position=(nz // 4, 15 * nx // 32), frequency=2.2e14, amplitude=10),
        ]
    initial_field_augmenters = [
        DampeningBorder(border_width=10, min_dampening=0.9, alpha=2.0),
        DampeningWall(position=(0, 600), size=(500, 30), min_dampening=0.90, reverse=True, alpha=3.0, gradient_axis='z'),
        DampeningWall(position=(520, 600), size=(20, 30), min_dampening=0.90, reverse=True, alpha=3.0, gradient_axis='z'),
        DampeningWall(position=(560, 600), size=(500, 30), min_dampening=0.90, reverse=True, alpha=3.0, gradient_axis='z'),
        ]
    
    sim = WaveSimulator(grid_size=(nx, nz), grid_spacing=(1e-7, 1e-7), scene_objects=scene_objects, initial_field_augmenters=initial_field_augmenters, time_scaler=1)
    vis = Visualiser(simulator=sim)
    key_handler = KeyHandler()

    while(True):
        sim.step()
        frame = sim.get_electric_field()
        vis.update( render_mode=key_handler.render_mode)

        key_handler.wait_key()

        if key_handler.quit:
            break

    vis.close()

if __name__ == "__main__":
    main()

