import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
import cv2
import os
from stable_baselines3.common.callbacks import BaseCallback
from datetime import datetime


class ArmEnv(gym.Env):
    def __init__(self, model_path):
        super(ArmEnv, self).__init__()
        
        # Load the MuJoCo model
        with open(model_path, 'r') as file:
            self.model_xml = file.read()
        self.model = mujoco.MjModel.from_xml_string(self.model_xml)
        self.data = mujoco.MjData(self.model)
        
        # Define action and observation space
        # Actions are the control signals for the motor
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Observations are the normalized positions of the arm tip and the ball, and velocity of all joints
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        
        # Simulation parameters
        self.timestep = 0.01
        self.max_steps = 700
        self.current_step = 0

    def reset(self, seed=None, options=None):
        # Reset the simulation
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.current_step = 0

        # Set the seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
        # Randomize the position of the ball
        radius = 0.36
        minimize_radius = 0.85 # lower to reduce randomness of x position
        ball_pos_x = float(np.random.uniform(-radius * minimize_radius, radius * minimize_radius))
        ball_pos_y = float(np.random.choice([1, 1]) * np.sqrt(radius ** 2 - ball_pos_x ** 2))
        self.model.body_pos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'ball')][0] = ball_pos_x
        self.model.body_pos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'ball')][1] = ball_pos_y
        
        # Return the initial observation
        return self._get_obs().astype(np.float32), {}

    def step(self, action):
        # Apply the action (control signal)
        self.data.ctrl[0] = action[0]
        
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
                       'angle': self._calc_angle(),
                       'obs': obs,
                       'current_step': self.current_step,
                       'arm_tip_pos': self.data.geom_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'arm_tip')],
                       'ball_pos': self.model.body_pos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'ball')],
                       'qvel': self.data.qvel[0]
                       }

        return obs.astype(np.float32), reward, terminated, truncated, information

    
    def _get_obs(self):
        # Get the positions of the arm tip and the ball, and velocity of all joints
        arm_tip_pos = self.data.geom_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'arm_tip')][:2]
        ball_pos = self.model.body_pos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'ball')][:2]
        dof_vel = self.data.qvel
        distance = self._calc_distance()
        
        # Normalize the observations
        arm_tip_pos = arm_tip_pos / np.array([0.15, 0.15])  # Assuming the range for arm_tip_pos is [-0.15, 0.15]
        ball_pos = ball_pos / np.array([0.36, 0.36])  # Assuming the range for ball_pos is [-0.36, 0.36]
        dof_vel = dof_vel / np.array([1.5])  # Assuming the range for dof_vel is [-1.5, 1.5]
        distance = (distance / (1.6 / 2)) - 1  # Normalize distance to be between -1 and 1
        
        # Concatenate the normalized positions to form the observation
        obs = np.concatenate([arm_tip_pos, ball_pos, dof_vel, [distance]])
        return obs

    def _calc_distance(self):
        arm_tip_pos = self.data.geom_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'arm_tip')]
        ball_pos = self.model.body_pos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'ball')]
        distance = np.linalg.norm(arm_tip_pos - ball_pos)
        return distance
    
    def _calc_angle(self):
        arm_tip_pos = self.data.geom_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'arm_tip')]
        ball_pos = self.model.body_pos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'ball')]
        angle_ball = float(np.arctan2(ball_pos[1], ball_pos[0]))
        angle_arm = float(np.arctan2(arm_tip_pos[1], arm_tip_pos[0]))
        angle_diff = angle_ball - angle_arm
        return angle_diff
    
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
        if self.current_step < 300:
            return False
        if self._calc_distance() <= 0.215 and abs(self.data.qvel[0]) < 0.29:
            return True
        if abs(self._calc_angle()) <= 0.1 and abs(self.data.qvel[0]) < 0.28:
            return True
        return False

    def _calculate_reward(self):
        distance = self._calc_distance()

        success_reward = 1 if self._is_success() else 0

        # Add a time-based penalty (more steps taken results in a lower reward)
        time_penalty = self.current_step

        # Discourage big velocities
        velocity_penalty = self.data.qvel[0]
        
        reward = - (distance ** 2) * 200 \
                 - (velocity_penalty ** 2) * 5 \
                 - time_penalty * 0.01 \
                 + success_reward * 10000
        
        return reward

    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            # Render the simulation and return an image frame
            width, height = 640, 480  # Set the desired width and height
            camera_name = 'my_camera'  # Set the camera name if needed
            renderer = mujoco.Renderer(self.model, width, height)
            renderer.update_scene(self.data, camera=camera_name)
            frame = renderer.render()
            renderer.close()
            return frame
        else:
            # Handle other rendering modes if needed
            raise ValueError


    def close(self):
        # Close the simulation (optional)
        pass



class RewardLoggerCallback(BaseCallback):
    def __init__(self, env, movie_save_freq=1000, model_save_freq=1000, verbose=2):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.env = env
        self.movie_save_freq = movie_save_freq
        self.model_save_freq = model_save_freq
        self.rewards = []
        self.timestamp = datetime.now().strftime("%m%d_%H%M%S")
        self.folder = os.path.join('training_result', f'{self.timestamp}')
        os.makedirs(self.folder, exist_ok=False)
        self.episode_rewards = []
        self.episode_avg_rewards = []
    

    def _on_step(self) -> bool:
        # Log the reward
        # self.rewards.append(self.locals['rewards'])
        self.episode_frames = []
        self.episode_names = []
        
        # Save frames at regular intervals
        if self.n_calls % self.movie_save_freq == 0:
            obs, _ = self.env.reset()  # Extract the observation from the tuple
            frames = []
            done = False
            i = 0
            episode_reward = 0
            while not done:
                action, _ = self.model.predict(obs)
                obs, reward, done, _, info = self.env.step(action)  # Extract the observation from the tuple
                frame = self.env.render(mode='rgb_array')               

                text_list = [
                f'Step: {info["current_step"]}',
                f'Action: {action[0]:.2f}',
                f'distance: {info["distance"]:.4f}',
                f'angle: {info["angle"]:.3f}',
                f'qvel: {info["qvel"]:.2f}',
                f'Arm Tip XYZ: {info["arm_tip_pos"][0]:.2f} {info["arm_tip_pos"][1]:.2f} {info["arm_tip_pos"][2]:.2f}',
                f'Ball XYZ: {info["ball_pos"][0]:.2f} {info["ball_pos"][1]:.2f} {info["ball_pos"][2]:.2f}',
                f'done: {done}',
                f'reward: {reward:.2f}',
                f'is_success: {info["is_success"]}'
                ]

                frames.append(add_text_frame(text_list, frame))
                episode_reward += reward
                i += 1
            self.episode_frames.append(frames)
            self.episode_names.append(self.n_calls)
            self._save_videos()
            self.episode_rewards.append(episode_reward)
            avg_reward = np.mean(self.episode_rewards)
            self.episode_avg_rewards.append(avg_reward)

            rewards_path = os.path.join(self.folder, f'rewards_{self.n_calls}.npy')
            np.save(rewards_path, self.episode_rewards)
            # print(f"Rewards saved at step {self.n_calls} to {rewards_path}")

        # Save the model and rewards at regular intervals
        if self.n_calls % self.model_save_freq == 0:
            model_path = os.path.join(self.folder, f'model_{self.n_calls}.zip')
            self.model.save(model_path)
            # print(f"Model saved at step {self.n_calls} to {model_path}")        
        return True

    def _save_videos(self) -> None:       
        # Save the episode frames to files
        for i, frames in enumerate(self.episode_frames):
            episode_name = self.episode_names[i]
            video_path = os.path.join(self.folder, f'ep_{episode_name}.mp4')
            height, width, _ = frames[0].shape
            out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
            for frame in frames:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            out.release()
    
    def _on_training_end(self) -> None:
        # Save the final model and rewards
        model_path = os.path.join(self.folder, f'model_training_end.zip')
        self.model.save(model_path)

        # Log the average reward at the end of training
        avg_reward = np.mean(self.episode_rewards)
        print(f"Average reward for the training session: {avg_reward:.2f}")
        rewards_path = os.path.join(self.folder, 'rewards_training_end.npy')
        np.save(rewards_path, self.episode_rewards)



def add_text_frame(texts, frame):
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    for i, t in enumerate(texts):
        cv2.putText(frame_bgr, t, (10, 30 + i * 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return frame_rgb



