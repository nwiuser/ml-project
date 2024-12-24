from params import  wall_penalty, goal_reward, step_penalty
import random
import numpy as np
from maze import Maze, maze_piece_cake, maze_medium

actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
def finish_episode(agent, maze, current_episode, train=True):
    current_state = maze.start_position
    done = False
    episode_reward = 0
    episode_step = 0
    path = [current_state] 

    while not done:
     
        action = agent.choose_action(current_state, current_episode)
        state_1 = current_state[0] + actions[action][0]
        state_2 = current_state[1] + actions[action][1]
        next_state = (state_1, state_2)

       
        if (
            next_state[0] < 0 or next_state[0] >= maze.width or
            next_state[1] < 0 or next_state[1] >= maze.height
        ):
          
            reward = wall_penalty
            next_state = current_state

        elif maze.maze[next_state[1]][next_state[0]] == 1:
         
            reward = wall_penalty
            next_state = current_state

        elif next_state == maze.goal_position:
   
            path.append(next_state)
            reward = goal_reward
            done = True

        else:
         
            path.append(next_state)
            reward = step_penalty

     
        episode_reward += reward
        episode_step += 1

     
        if train:
            agent.update_q_table(current_state, action, reward, next_state)


        current_state = next_state

    return episode_reward, episode_step, path


# def fit_q_learning_agent(maze, num_episodes=1000, alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.995):

#   best_wall_penalty = 0
#   best_step_penalty = 0
#   best_goal_reward = 0
#   best_q_table = None
#   best_average_reward = -float('inf')


#   wall_penalty_range = np.arange(-10, -1, 1)  
#   step_penalty_range = np.arange(-0.1, -0.01, 0.01)  
#   goal_reward_range = np.arange(10, 50, 5)

#   for wp in wall_penalty_range:
#     for sp in step_penalty_range:
#       for gr in goal_reward_range:
#         q_table = np.zeros((maze.shape[0], maze.shape[1], 4))  


#         average_reward = 0
#         for episode in range(num_episodes):
#           state = np.where(maze == 2) 
#           state = (state[0][0], state[1][0])
#           total_reward = 0

#           while True:
           
#             if random.uniform(0, 1) < epsilon:
#               action = random.randint(0, 3)  
#             else:
#               action = np.argmax(q_table[state[0], state[1], :])  

#             next_state, reward = take_action(maze, state, action, wp, sp, gr)

#             q_table[state[0], state[1], action] = q_table[state[0], state[1], action] + \
#                                                 alpha * (reward + gamma * np.max(q_table[next_state[0], next_state[1], :]) - 
#                                                         q_table[state[0], state[1], action])

#             total_reward += reward
#             state = next_state

#             if maze[state] == 3: 
#               break

#           average_reward += total_reward

#           epsilon *= epsilon_decay

#         average_reward /= num_episodes

#         if average_reward > best_average_reward:
#           best_average_reward = average_reward
#           best_wall_penalty = wp
#           best_step_penalty = sp
#           best_goal_reward = gr
#           best_q_table = q_table.copy()

#   return best_wall_penalty, best_step_penalty, best_goal_reward, best_q_table

# def take_action(maze, state, action, wall_penalty, step_penalty, goal_reward):


#   row, col = state
#   if action == 0:  
#     next_row = row - 1
#   elif action == 1:  
#     next_col = col + 1
#   elif action == 2:  
#     next_row = row + 1
#   elif action == 3: 
#     next_col = col - 1

#   if next_row < 0 or next_row >= maze.shape[0] or next_col < 0 or next_col >= maze.shape[1] or maze[next_row, next_col] == 1:
#     next_state = state
#     reward = wall_penalty
#   else:
#     next_state = (next_row, next_col)
#     reward = step_penalty

#   if maze[next_state] == 3:
#     reward += goal_reward

#   return next_state, reward

# best_wp, best_sp, best_gr, best_q_table = fit_q_learning_agent(maze_piece_cake)
# print(f"Best wall penalty: {best_wp}")
# print(f"Best step penalty: {best_sp}")
# print(f"Best goal reward: {best_gr}")
