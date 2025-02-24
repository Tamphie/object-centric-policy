import wandb
import numpy as np
import torch
import tqdm


from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
import diffusion_policy_3d.common.logger_util as logger_util
from termcolor import cprint


class RlbenchRunner(BaseRunner):
    def __init__(self,
                 output_dir,
                 eval_episodes=20,
                 max_steps=200,
                 n_obs_steps=8,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 render_size=84,
                 tqdm_interval_sec=5.0,
                 task_name=None,
                 use_point_crop=True,
                 ):
        super().__init__(output_dir)
        
        from rlbench.backend.utils import task_file_to_task_class
        task_name = task_file_to_task_class(task_name)
        self.task_name = task_name
        steps_per_render = max(10 // fps, 1)

        def env_fn():
            from rlbench.environment import Environment
            from rlbench.action_modes.action_mode import MoveArmThenGripper
            from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK,JointPosition,EndEffectorPoseViaPlanning
            from rlbench.action_modes.gripper_action_modes import Discrete
            from rlbench.observation_config import ObservationConfig
            obs_config = ObservationConfig()
            obs_config.set_all(True)
            return Environment(
                action_mode=MoveArmThenGripper(
                #action_shape 7; 1
                arm_action_mode=EndEffectorPoseViaIK(), gripper_action_mode=Discrete()),
                obs_config=obs_config,
                headless=False)
            
        self.eval_episodes = eval_episodes
        self.env = env_fn()
        self.env.launch()
        self.task_env = self.env.get_task(task_name)
        self.env = self.task_env
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)
    # Convert RLBench observation object to a dictionary
    #TODO agent_pos
    def convert_rlbench_obs(self, obs):
        gripper_open = np.array(obs.gripper_open).reshape(1)
        joint_positions = np.array(obs.joint_positions)
        qpos = np.concatenate([joint_positions,gripper_open])
        qpos = np.array(qpos)
        qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0).cpu().numpy()
        print(f"Shape of qpos at timestep during inference : {qpos.shape}")
        np_obs_dict = {
            "point_cloud": np.array(obs.pcd_from_mesh),   # Convert point cloud to NumPy array
            "agent_pos": qpos, # Use joint positions as robot state
            # "joint_velocities": np.array(obs.joint_velocities),
            # "gripper_open": np.array([obs.gripper_open]), # Convert boolean to NumPy array
            # "image": np.array(obs.wrist_camera_rgb), # Wrist camera image
            # "task_low_dim_state": np.array(obs.task_low_dim_state), # Task-specific state
        }
        print(f"Shape of obs.pcd_from_mesh: {np.array(obs.pcd_from_mesh).shape}")

        return np_obs_dict

    def run(self, policy: BasePolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env

        all_goal_achieved = []
        all_success_rates = []
        


        for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in Adroit {self.task_name} Pointcloud Env",
                                     leave=False, mininterval=self.tqdm_interval_sec):
                
            # start rollout
          
            policy.reset()

            done = False
            num_goal_achieved = 0
            actual_step_count = 0
            while not done:
                # create obs dict
                if actual_step_count == 0:
                    descriptions, obs = env.reset()
                    # obs = env.get_observation()
                    print(f"reset successfully")
                else:
                    obs = env.get_observation()
            
                np_obs_dict = self.convert_rlbench_obs(obs)
                # device transfer
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x).to(
                                          device=device))

                # run policy
                with torch.no_grad():
                    obs_dict_input = {}  # flush unused keys
                    obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
                    print(f"Shape of obs_dict_input['point_cloud']: {obs_dict_input['point_cloud'].shape}")
                    obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
                    action_dict = policy.predict_action(obs_dict_input)
                    

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())
                
                action = np_action_dict['action'].squeeze(0)
                action = np.mean(action, axis=0)  # (3,8) → (8,)
                print(f"Action shape: {action.shape}")
                # step env
                step_result = env.step(action)
                if len(step_result) == 3:  # Handle missing info
                    obs, reward, done = step_result
                    info = {"goal_achieved": 0}  # Default value
                else:
                    obs, reward, done, info = step_result  # Use provided info
                # all_goal_achieved.append(info['goal_achieved']
                # num_goal_achieved += np.sum(info['goal_achieved'])
                done = np.all(done)
                actual_step_count += 1

            all_success_rates.append(info['goal_achieved'])
            all_goal_achieved.append(num_goal_achieved)


        # log
        log_data = dict()
        

        log_data['mean_n_goal_achieved'] = np.mean(all_goal_achieved)
        log_data['mean_success_rates'] = np.mean(all_success_rates)

        log_data['test_mean_score'] = np.mean(all_success_rates)

        cprint(f"test_mean_score: {np.mean(all_success_rates)}", 'green')

        self.logger_util_test.record(np.mean(all_success_rates))
        self.logger_util_test10.record(np.mean(all_success_rates))
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()

        videos = env.env.get_video()
        if len(videos.shape) == 5:
            videos = videos[:, 0]  # select first frame
        videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")
        log_data[f'sim_video_eval'] = videos_wandb

        # clear out video buffer
        _ = env.reset()
        # clear memory
        videos = None
        del env

        return log_data
