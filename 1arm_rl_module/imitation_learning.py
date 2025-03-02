import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.utils import obs_as_tensor
import torch as th

from arm_env import ArmEnv

class ExpertDataset:
    def __init__(self, observations, actions):
        self.observations = observations
        self.actions = actions

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]

def collect_expert_data(env, model, num_episodes=10):
    observations = []
    actions = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            observations.append(obs)
            actions.append(action)
            obs, _, done, _, _ = env.step(action)
    return np.array(observations), np.array(actions)

def train_behavior_cloning(env, expert_dataset, num_epochs=10, batch_size=32):
    policy = ActorCriticPolicy(env.observation_space, env.action_space, lr_schedule=lambda _: 1e-3)
    optimizer = th.optim.Adam(policy.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        for i in range(0, len(expert_dataset), batch_size):
            batch_obs, batch_actions = expert_dataset[i:i+batch_size]
            batch_obs = obs_as_tensor(batch_obs, policy.device)
            batch_actions = obs_as_tensor(batch_actions, policy.device)

            # Forward pass
            distribution, value = policy(batch_obs)
            log_prob = distribution.log_prob(batch_actions)

            # Compute loss
            loss = -log_prob.mean()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

    return policy

if __name__ == "__main__":
    # Create the custom environment
    env = ArmEnv(model_path='myrobot.xml')
    env = DummyVecEnv([lambda: env])

    # Load the expert model
    expert_model = PPO.load("ppo_arm")

    # Collect expert data
    observations, actions = collect_expert_data(env, expert_model, num_episodes=10)
    expert_dataset = ExpertDataset(observations, actions)

    # Train the behavior cloning model
    policy = train_behavior_cloning(env, expert_dataset, num_epochs=10, batch_size=32)

    # Save the trained policy
    th.save(policy.state_dict(), "behavior_cloning_policy.pth")