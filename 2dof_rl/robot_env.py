import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
import util



class ArmEnv(gym.Env):
    def __init__(self, model_path):
        super(ArmEnv, self).__init__()
        
        # Load the MuJoCo model
        with open(model_path, 'r') as file:
            self.model_xml = file.read()
        self.model = mujoco.MjModel.from_xml_string(self.model_xml)
        self.data = mujoco.MjData(self.model)
        self.json_vars = util.load_json_variables()
        
        # Define action and observation space
        # Actions are the control signals for the motor
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # Observations are the normalized positions of the arm tip and the ball, and velocity of all joints
        self.observation_space = spaces.Box(low=-1, high=1, shape=(13,), dtype=np.float32)
        
        # Simulation parameters
        self.timestep = 0.01
        self.max_steps = 2000
        self.current_step = 0

    def reset(self, seed=None, options=None):
        # Reset the simulation
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.current_step = 0
  
        util.randomize_ball_position(self.model, self.json_vars, seed=seed)
        
        # Return the initial observation
        return self._get_obs().astype(np.float32), {}

    def step(self, action):
        # Apply the action (control signal)
        self.data.ctrl[0] = action[0]
        self.data.ctrl[1] = action[1]
        
        # Step the simulation
        mujoco.mj_step(self.model, self.data)
        
        # Get the observation
        obs = self._get_obs()
        
        # Calculate the reward
        reward = self._calculate_reward()
        
        # Check if the episode is done
        self.current_step += 1
        
        if self._is_terminal():
            terminated = True
        else:
            terminated = False

        truncated = False  # You can add your own condition for truncation if needed
        
        information = {'is_success': self._is_success(),
                       'distance': self._calc_distance(),
                       'obs': obs,
                       'current_step': self.current_step,
                       'arm2_tip_pos': self.data.site_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'arm2_tip')],
                       'ball_pos': self.model.body_pos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'ball')],
                       'qvel': self.data.qvel
                       }

        return obs.astype(np.float32), reward, terminated, truncated, information

    
    def _get_obs(self):
        # Get the positions of the arm tip and the ball, and velocity of all joints
        arm1_pos = self.data.geom_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'arm1_geom')][:2]
        arm2_pos = self.data.geom_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'arm2_geom')][:2]
        arm2_tip_pos = self.data.site_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'arm2_tip')][:2]
        ball_pos = self.model.body_pos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'ball')][:2]
        qvel = self.data.qvel # 2 joints
        qacc = self.data.qacc # 2 joints
        distance = self._calc_distance()
        
        # Normalize the observations
        arm1_pos = arm1_pos / np.array([200, 200])
        arm2_pos = arm2_pos / np.array([200, 200])
        arm2_tip_pos = arm2_tip_pos / np.array([300, 300])
        ball_pos = ball_pos / np.array([300, 300])
        qvel = qvel / np.array([200, 200])
        qacc = qacc / np.array([10, 10])
        distance = (distance / (300 / 2)) - 1  # Normalize distance to be between -1 and 1
        
        # Concatenate the normalized positions to form the observation
        obs = np.concatenate([arm1_pos, arm2_pos, arm2_tip_pos, ball_pos, qvel, qacc, [distance]]) #13 observations
        return obs

    def _calc_distance(self):
        arm2_tip_pos = self.data.site_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'arm2_tip')]
        ball_pos = self.model.body_pos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'ball')]
        distance = np.linalg.norm(arm2_tip_pos - ball_pos)
        return distance
        
    def _is_terminal(self):
        # Check if the episode is done
        if self.current_step >= self.max_steps:
            return True
        if self._is_success():
            return True
        return False
    
    def _is_success(self):
        # Check if the episode is successful
        # if too early, maybe just a chance that the ball start close to the arm tip
        if self.current_step < 100:
            return False
        if abs(self.data.qvel[0]) > 0.5:
            return False
        if abs(self.data.qvel[1]) > 0.5:
            return False
        if self._calc_distance() <= 15:
            return True
        return False

    def _calculate_reward(self):
        distance = self._calc_distance()

        success_reward = 1 if self._is_success() else 0

        # Add a time-based penalty (more steps taken results in a lower reward)
        time_penalty = self.current_step

        # # Discourage big velocities
        velocity1_penalty = self.data.qvel[0] ** 2
        velocity2_penalty = self.data.qvel[0] ** 2
        
        reward = - distance * 0.1 \
                 - velocity1_penalty * 0.1 \
                 - velocity2_penalty * 0.1 \
                 - time_penalty * 0.01 \
                 + success_reward * 1000
        
        return reward

    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            # Render the simulation and return an image frame
            width, height = 480, 480  # Set the desired width and height
            renderer = mujoco.Renderer(self.model, width, height)
            renderer.update_scene(self.data, camera='my_camera')
            frame = renderer.render()
            renderer.close()
            return frame
        else:
            # Handle other rendering modes if needed
            raise ValueError


    def close(self):
        # Close the simulation (optional)
        pass






