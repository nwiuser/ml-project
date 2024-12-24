from qlearn import QLearningAgent
# from montecarlo import MonteCarloAgent
# from sarsa import SARSAAgent
from maze import Maze, maze_piece_cake, maze_fire, maze_medium


goal_reward = 10
wall_penalty = -50
step_penalty = 1

# maze_piece_cake.goal_position = (9,9)
# maze_medium.goal_position = (20,20)
# maze_fire.goal_position = (43,35)

maze = Maze(maze_piece_cake, (0,0), (9,9))
agent_qlearn = QLearningAgent(maze=maze)

# agent_montecarlo = MonteCarloAgent(maze=maze)

# agent_sarsa = SARSAAgent(maze=maze)