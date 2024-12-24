import numpy as np
from maze import Maze, maze_piece_cake, maze_fire, maze_medium

actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

class MonteCarloAgent:
    def __init__(self, maze, learning_rate=0.1, discount_factor=0.9, exploration_start=1.0, exploration_end=0.01, num_episodes=100):
        self.q_table = np.zeros((maze.height, maze.width, len(actions))) 
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_start = exploration_start
        self.exploration_end = exploration_end
        self.num_episodes = num_episodes

    def get_exploration_rate(self, current_episode):
     
        return self.exploration_start + (self.exploration_end / self.exploration_start) ** (current_episode / self.num_episodes)

    def choose_action(self, state, current_episode):
        
        exploration_rate = self.get_exploration_rate(current_episode)
        if np.random.rand() < exploration_rate:
            return np.random.choice(len(actions))  
        else:
            return np.argmax(self.q_table[state[0], state[1]])  

    def update_q_table(self, states, rewards):
    
        G = 0
        
        for t in range(len(states) - 1, -1, -1):
            G = rewards[t] + self.discount_factor * G  
            state = states[t]
            action = states[t][2] 
            
            self.q_table[state[0], state[1], action] = (self.q_table[state[0], state[1], action] + G) / 2  

