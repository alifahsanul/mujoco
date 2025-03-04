import os
import cv2
import numpy as np
from datetime import datetime
from stable_baselines3.common.callbacks import BaseCallback
import util


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

                get_frame = False
                if done:
                    get_frame = True
                if i % 10==0:
                    get_frame = True


                if get_frame:
                    frame = self.env.render(mode='rgb_array')               

                    text_list = [
                    f'Step: {info["current_step"]}',
                    f'Action: {action[0]:.2f} {action[1]:.2f}',
                    f'distance: {info["distance"]:.4f}',
                    f'qvel: {info["qvel"][0]:.2f} {info["qvel"][1]:.2f}',
                    f'Arm Tip XYZ: {info["arm2_tip_pos"][0]:.2f} {info["arm2_tip_pos"][1]:.2f} {info["arm2_tip_pos"][2]:.2f}',
                    f'Ball XYZ: {info["ball_pos"][0]:.2f} {info["ball_pos"][1]:.2f} {info["ball_pos"][2]:.2f}',
                    f'done: {done}',
                    f'reward: {reward:.2f}',
                    f'is_success: {info["is_success"]}'
                    ]

                    frames.append(util.add_text_frame(text_list, frame))
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
            out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 3, (width, height))
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





def linear_schedule(initial_value):
    """
    Linear learning rate schedule.
    :param initial_value: (float) Initial learning rate.
    :return: (function) Schedule that computes the current learning rate.
    """
    def func(progress_remaining):
        """
        Progress will decrease from 1 (beginning) to 0.
        :param progress_remaining: (float) Progress remaining (from 1 to 0).
        :return: (float) Current learning rate.
        """
        return progress_remaining * initial_value
    return func

