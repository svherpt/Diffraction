from src.simulator import WaveSimulator
from src.visualiser import Visualiser
from src.scene_objects import PointSource, LineSource
from src.initial_field_augmenters import DampeningWall, DampeningBorder
from src.key_handler import KeyHandler
import cv2
import cupy as cp


def main():

    nx, nz = 1080, 1920

    scene_objects = [
        # PointSource(position=(nz // 4, 17 * nx // 32), frequency=2.2e14, amplitude=10),
        # PointSource(position=(nz // 4, 16 * nx // 32), frequency=2.2e14, amplitude=10),
        # PointSource(position=(nz // 4, 15 * nx // 32), frequency=2.2e14, amplitude=10),
        LineSource(start_position=(40, 20), size=(1, nx-40), frequency=2.2e14, amplitude=10),
        ]

    wall_min_dampening = 0.80
    wall_alpha = 2.0

    initial_field_augmenters = [
        DampeningBorder(border_width=20, min_dampening=0.9, alpha=2.0),
        DampeningWall(position=(0, 600), size=(490, 30), min_dampening=wall_min_dampening, alpha=wall_alpha),
        DampeningWall(position=(510, 600), size=(20, 30), min_dampening=wall_min_dampening, alpha=wall_alpha),
        DampeningWall(position=(550, 600), size=(20, 30), min_dampening=wall_min_dampening, alpha=wall_alpha),
        DampeningWall(position=(590, 600), size=(490, 30), min_dampening=wall_min_dampening, alpha=wall_alpha),
        ]
    
    sim = WaveSimulator(grid_size=(nx, nz), grid_spacing=(1e-7, 1e-7), scene_objects=scene_objects, initial_field_augmenters=initial_field_augmenters, time_scaler=1)
    vis = Visualiser(simulator=sim)
    key_handler = KeyHandler()

    while(True):
        event = key_handler.wait_key()

        sim.update(event)
        frame = sim.get_electric_field()
        vis.update(render_mode=key_handler.render_mode)


        if key_handler.quit:
            break

    vis.close()

if __name__ == "__main__":
    main()

