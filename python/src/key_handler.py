import cv2

class KeyHandler:
    def __init__(self):
        self.render_mode = 'electric'
        self.quit = False

    def wait_key(self):
        key = cv2.waitKey(1) & 0xFF

        rendering_modes = {ord('d'): 'dampening', ord('f'): 'field', ord('e'): 'electric', ord('r'): 'refraction'}

        if key == ord('q'):
            self.quit = True

        if key in rendering_modes:
            self.render_mode = rendering_modes[key]

        if key == ord('s'):
            return {"type": "skip", "num_steps": 1000}

        return {"type": "normal"}