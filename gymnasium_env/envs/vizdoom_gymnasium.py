from typing import Optional, Dict
import vizdoom
import numpy as np
import itertools
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import cv2
from utils.image import img_write_text
import os


def get_resolution_format_type(resolution: tuple[int, int]):
    resolution_map = {
        (160, 120): vizdoom.ScreenResolution.RES_160X120,
        (200, 125): vizdoom.ScreenResolution.RES_200X125,
        (200, 150): vizdoom.ScreenResolution.RES_200X150,
        (256, 144): vizdoom.ScreenResolution.RES_256X144,
        (256, 160): vizdoom.ScreenResolution.RES_256X160,
        (256, 192): vizdoom.ScreenResolution.RES_256X192,
        (320, 180): vizdoom.ScreenResolution.RES_320X180,
        (320, 200): vizdoom.ScreenResolution.RES_320X200,
        (320, 240): vizdoom.ScreenResolution.RES_320X240,
        (320, 256): vizdoom.ScreenResolution.RES_320X256,
        (400, 225): vizdoom.ScreenResolution.RES_400X225,
        (400, 250): vizdoom.ScreenResolution.RES_400X250,
        (400, 300): vizdoom.ScreenResolution.RES_400X300,
        (512, 288): vizdoom.ScreenResolution.RES_512X288,
        (512, 320): vizdoom.ScreenResolution.RES_512X320,
        (512, 384): vizdoom.ScreenResolution.RES_512X384,
        (640, 360): vizdoom.ScreenResolution.RES_640X360,
        (640, 400): vizdoom.ScreenResolution.RES_640X400,
        (640, 480): vizdoom.ScreenResolution.RES_640X480,
        (800, 450): vizdoom.ScreenResolution.RES_800X450,
        (800, 500): vizdoom.ScreenResolution.RES_800X500,
        (800, 600): vizdoom.ScreenResolution.RES_800X600,
        (1024, 576): vizdoom.ScreenResolution.RES_1024X576,
        (1024, 640): vizdoom.ScreenResolution.RES_1024X640,
        (1024, 768): vizdoom.ScreenResolution.RES_1024X768,
        (1280, 720): vizdoom.ScreenResolution.RES_1280X720,
        (1280, 800): vizdoom.ScreenResolution.RES_1280X800,
        (1280, 960): vizdoom.ScreenResolution.RES_1280X960,
        (1280, 1024): vizdoom.ScreenResolution.RES_1280X1024,
        (1400, 787): vizdoom.ScreenResolution.RES_1400X787,
        (1400, 875): vizdoom.ScreenResolution.RES_1400X875,
        (1400, 1050): vizdoom.ScreenResolution.RES_1400X1050,
        (1600, 900): vizdoom.ScreenResolution.RES_1600X900,
        (1600, 1000): vizdoom.ScreenResolution.RES_1600X1000,
        (1600, 1200): vizdoom.ScreenResolution.RES_1600X1200,
        (1920, 1080): vizdoom.ScreenResolution.RES_1920X1080,
    }

    ret = resolution_map.get(resolution)

    if ret:
        return ret
    else:
        x, y = resolution
        raise ValueError(f"Resolution not supported: {x}x{y}")


# https://gymnasium.farama.org/introduction/create_custom_env/
class VizdoomGymnasiumEnv(Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 35}

    # Function that is called when we start the env
    def __init__(
        self,
        out_file: str = "./artifacts/renders/vizdoom_out.mp4",
        resolution: tuple[int, int] = (640, 480),
        fps: int = 35,
        window_visible: bool = True,
        frame_skip: int = 1,
        render_mode: str = "rgb_array",
    ):
        # Inherit from Env
        super().__init__()
        # Set up the game
        self.game = vizdoom.DoomGame()
        self.game.load_config(
            os.path.join(vizdoom.scenarios_path, "basic.cfg")
        )
        self.game.set_mode(vizdoom.Mode.PLAYER)
        self.game.set_screen_format(vizdoom.ScreenFormat.BGR24)

        self.resolution = resolution
        resolution_format = vizdoom.ScreenResolution.RES_640X480
        try:
            resolution_format = get_resolution_format_type(resolution)
        except ValueError:
            x, y = resolution
            print(f"Invalid resolution {x}x{y}, defaulting to 640x480")
            self.resolution = (640, 480)

        self.game.set_screen_resolution(resolution_format)

        self.frame_skip = 1 if render_mode is not None else frame_skip
        self.fps = fps

        if not window_visible:
            self.game.set_window_visible(False)
        else:
            self.game.set_window_visible(True)

        assert (
            render_mode is None or render_mode in self.metadata["render_modes"]
        )
        self.render_mode = render_mode

        # Start the game
        self.game.init()

        # Create the action space and observation space
        # 200x150 RGB
        self.observation_space = Box(
            low=0, high=255, shape=(150, 200, 3), dtype=np.uint8
        )
        # Create an action list
        # Ex: For basic.cfg we have 3 buttons so it will create this:
        # [[0,0,0], [0,0,1], [0,1,0], [0,1,1],
        # [1,0,0], [1,0,1], [1,1,0], [1,1,1]]
        n = self.game.get_available_buttons_size()
        self.actions = [list(a) for a in itertools.product([0, 1], repeat=n)]
        self.action_space = Discrete(len(self.actions))
        self.last_ammo = 0

    def get_observation(self):
        game_state = self.game.get_state()
        if game_state:
            state = game_state.screen_buffer
            np.moveaxis(state, 0, -1)
            # state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
            shape = self.observation_space.shape
            state = cv2.resize(
                state, shape[1::-1], interpolation=cv2.INTER_CUBIC
            )
            state = np.reshape(state, shape)
            return state
        else:
            return np.zeros(self.observation_space.shape)

    def get_info(self) -> Dict[str, int]:
        game_state = self.game.get_state()
        if game_state:
            if game_state.game_variables is not None:
                ammo = game_state.game_variables[0]
                self.last_ammo = ammo
            else:
                # VizDoom doesn't give this variable after episode termination
                # so we just save the last one here
                ammo = self.last_ammo
            info = {"ammo": ammo}
            return info
        else:
            return {"ammo": self.last_ammo}

    # This is how we take a step in the environment
    def step(self, action: int):
        reward = self.game.make_action(self.actions[action], self.frame_skip)

        observation = self.get_observation()

        info = self.get_info()
        terminated = self.game.is_episode_finished()
        truncated = False

        # If no ammo is used punish, attempt to prevent trashing
        if terminated and (info.get("ammo") >= 50):
            reward -= 6

        return observation, reward, terminated, truncated, info

    # Define how to render the game or environment
    def render(self):
        if self.render_mode == "rgb_array":
            state = self.game.get_state()
            if state is not None:
                buffer = state.screen_buffer
                buffer = img_write_text(
                    buffer,
                    (8, 8),
                    f"Total score: {self.game.get_total_reward()}",
                )
                buffer = img_write_text(
                    buffer,
                    (8, 64),
                    f"Last Reward: {self.game.get_last_reward()}",
                )
                # TODO: Find a way to render bgr on gymnasium
                buffer = cv2.cvtColor(buffer, cv2.COLOR_BGRA2RGB)

                return buffer
            # else:
            # TODO: find out why this doesn't work
            # x, y = self.resolution
            # return np.zeros((x, y, 3), dtype=np.uint8)
        return None

    # What happens when we start a new game
    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ):
        super().reset(seed=seed)

        if seed is not None:
            self.game.set_seed(seed)
        self.game.new_episode()
        observation = self.get_observation()
        info = self.get_info()

        return observation, info

    # Call to close down the game
    def close(self):
        self.game.close()
