import numpy as np
from maze import Maze, maze_piece_cake, maze_fire, maze_medium

actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
class QLearningAgent:
  def __init__(self, maze, learning_rate=0.1, discount_factor=0.9, exploration_start=1.0, exploration_end=0.01, num_episodes=100):
    self.q_table = np.zeros((maze.height, maze.width, len(actions)))
    self.learning_rate = learning_rate
    self.discount_factor = discount_factor
    self.exploration_start = exploration_start
    self.exploration_end = exploration_end
    self.num_episodes = num_episodes

  def get_exploration_rate(self, current_episode):
    exploration_rate = self.exploration_start + (self.exploration_end / self.exploration_start) ** (current_episode / self.num_episodes)
    return exploration_rate

  def choose_action(self, state, current_episode):
    exploration_rate = self.get_exploration_rate(current_episode)
    if np.random.rand() < exploration_rate:
      return np.random.choice(len(actions))
    else:
      return np.argmax(self.q_table[state])

  def update_q_table(self, state, action, reward, next_state):
    try:
        if next_state[0] >= self.q_table.shape[0] or next_state[1] >= self.q_table.shape[1]:
            print(f"Invalid next state: {next_state}. Skipping update.")
            return
        
        best_next_action = np.argmax(self.q_table[next_state[0], next_state[1]])
        
        current_q_value = self.q_table[state[0], state[1], action]
     
        new_value = current_q_value + self.learning_rate * (
            reward + self.discount_factor * self.q_table[next_state[0], next_state[1], best_next_action] - current_q_value
        )
  
        self.q_table[state[0], state[1], action] = new_value
    except IndexError as e:
        print(f"IndexError in update_q_table: {e}")
        print(f"State: {state}, Next State: {next_state}")
