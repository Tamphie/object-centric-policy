import gym
import numpy as np
from termcolor import cprint


class SimpleVideoRecordingWrapper(gym.Wrapper):
    def __init__(self, 
            env, 
            mode='rgb_array',
            steps_per_render=1,
        ):
        """
        When file_path is None, don't record.
        """
        super().__init__(env)
        
        self.mode = mode
        self.steps_per_render = steps_per_render

        self.step_count = 0

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self.frames = list()
        # print(f"🔍 Debug: self.env = {self.env}")
        # print(f"🔍 Debug: Type of self.env = {type(self.env)}")
        # print(f"🔍 Debug: Available methods in self.env = {dir(self.env)}")

        # # Check if render() exists and what it accepts
        # if hasattr(self.env, "render"):
        #     import inspect
        #     print(f"🔍 Debug: render() signature = {inspect.signature(self.env.render)}")
        # else:
        #     print("❌ self.env does NOT have a render() method!")
        # frame = self.env.render(mode=self.mode)
        frame = self.env.render()
        if frame: 
            assert frame.dtype == np.uint8
        self.frames.append(frame)
        self.step_count = 1
        return obs
    
    def step(self, action):
        result = super().step(action)
        self.step_count += 1
        
        # frame = self.env.render(mode=self.mode)
        frame = self.env.render()
        if frame: 
            assert frame.dtype == np.uint8
        self.frames.append(frame)
        
        return result
    
    def get_video(self):
        video = np.stack(self.frames, axis=0) # (T, H, W, C)
        # to store as mp4 in wandb, we need (T, H, W, C) -> (T, C, H, W)
        video = video.transpose(0, 3, 1, 2)
        return video

