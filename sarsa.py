import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import time
from maze import Maze, maze_fire, maze_medium, maze_piece_cake
from params import wall_penalty, goal_reward, step_penalty


actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]


class SARSAAgent:
    def __init__(self, maze, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.q_table = np.zeros((maze.height, maze.width, len(actions)))
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.actions = actions

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(len(self.actions))
        else:
            return np.argmax(self.q_table[state[0], state[1]])

    def update(self, state, action, reward, next_state, next_action):
        q_predict = self.q_table[state[0], state[1], action]
        q_target = reward + self.gamma * self.q_table[next_state[0], next_state[1], next_action]
        self.q_table[state[0], state[1], action] += self.lr * (q_target - q_predict)



def finish_episode_sarsa(agent, maze, current_episode, train=True):
   
    current_state = maze.start_position
    done = False
    episode_reward = 0
    episode_step = 0
    path = [current_state] 

    
    action = agent.choose_action(current_state, current_episode)

    while not done:
    
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

    
        next_action = agent.choose_action(next_state, current_episode)

     
        if train:
            agent.update(current_state, action, reward, next_state, next_action)

      
        current_state = next_state
        action = next_action

    return episode_reward, episode_step, path



def train_agent_sarsa(agent, maze, num_episodes=100):
    st.title("Entraînement de l'Agent SARSA - Visualisation en Temps Réel")

    episode_rewards = []
    episode_steps = []

    reward_chart = st.empty()
    steps_chart = st.empty()
    stats = st.empty()
    progress_bar = st.progress(0)

    start_time = time.time()

    for episode in range(num_episodes):
        episode_reward, episode_step, path = finish_episode_sarsa(agent, maze,episode, train=True)

        episode_rewards.append(episode_reward)
        episode_steps.append(episode_step)

        average_reward = np.mean(episode_rewards)
        average_steps = np.mean(episode_steps)
        elapsed_time = time.time() - start_time
        formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

        stats.markdown(f"""
        **Épisode : {episode + 1}/{num_episodes}**
        - Récompense cumulée : {episode_reward}
        - Pas pris : {episode_step}
        - Récompense moyenne : {average_reward:.2f}
        - Nombre moyen de pas : {average_steps:.2f}
        - Temps écoulé : {formatted_time}
        """)

        with reward_chart.container():
            fig, ax = plt.subplots()
            ax.plot(episode_rewards, label="Récompense par épisode", color="blue")
            ax.set_title("Récompenses cumulées par épisode")
            ax.set_xlabel("Épisode")
            ax.set_ylabel("Récompense cumulée")
            ax.legend()
            st.pyplot(fig)

        with steps_chart.container():
            fig, ax = plt.subplots()
            ax.plot(episode_steps, label="Étapes par épisode", color="green")
            ax.set_title("Nombre d'étapes par épisode")
            ax.set_xlabel("Épisode")
            ax.set_ylabel("Étapes prises")
            ax.legend()
            st.pyplot(fig)

        progress_bar.progress((episode + 1) / num_episodes)

    total_time = time.time() - start_time
    formatted_total_time = time.strftime("%H:%M:%S", time.gmtime(total_time))

    st.success("Entraînement terminé ! 🎉")
    st.write(f"Récompense moyenne : {average_reward:.2f}")
    st.write(f"Nombre moyen de pas : {average_steps:.2f}")
    st.write(f"Temps total d'entraînement : {formatted_total_time}")


maze = Maze(maze_piece_cake, (0,0), (9,9))
agent_sarsa = SARSAAgent(maze)


train_agent_sarsa(agent_sarsa, maze, num_episodes=100)
