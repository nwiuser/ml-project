import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import time
import random
from qlearn import QLearningAgent, actions
from maze import Maze, maze_fire, maze_medium, maze_piece_cake
from fonctions import finish_episode
import params

# <------- les fonctions ------->

actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def test_agent(agent, maze, num_episodes=1):
  episode_reward, episode_step, path = finish_episode(agent, maze, 1, train=False)

  print("Learned Path:")
  for row, col in path:
    print(f"({row}, {col})-> ", end='')
  print('Goal!')

  print("Number of steps : ", episode_step)
  print("Total reward : ", episode_reward)

 
  if plt.gcf().get_axes():
    plt.cla()


  plt.figure(figsize=(10,10))
  plt.imshow(maze.maze, cmap='gray')

  if np.array_equal(maze.maze, maze_piece_cake):
      size = (5, 5)
      fontsize = 20
  elif np.array_equal(maze.maze, maze_medium):
      size = (10, 10)
      fontsize = 15
  elif np.array_equal(maze.maze, maze_fire):
      size = (15, 15)
      fontsize = 10
  else:
      size = (5, 5)
      fontsize = 15

 
  plt.text(maze.start_position[0], maze.start_position[1], 'S', ha='center', va='center', color='red', fontsize=fontsize)
  plt.text(maze.goal_position[0], maze.goal_position[1], 'G', ha='center', va='center', color='green', fontsize=fontsize)

  
  for position in path:
    plt.text(position[0], position[1], '#', va='center', color='yellow', fontsize=fontsize)

  
  plt.xticks([]), plt.yticks([])
  plt.grid(color='black', linewidth=2)
  plt.show()

  return episode_reward, episode_step, path