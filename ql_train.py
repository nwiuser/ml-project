import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import time
import random
from qlearn import QLearningAgent, actions
from maze import Maze, maze_fire, maze_medium, maze_piece_cake
from fonctions import finish_episode
from params import agent_qlearn, maze

# <------- les fonctions ------->

actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def train_agent(agent, maze, num_episodes=100):
    st.title("Entraînement de l'Agent - Visualisation en Temps Réel")

    episode_rewards = []
    episode_steps = []

    reward_chart = st.empty()
    steps_chart = st.empty()
    stats = st.empty()
    progress_bar = st.progress(0)


    start_time = time.time()

    for episode in range(num_episodes):
  
        episode_reward, episode_step, path = finish_episode(agent, maze, episode, train=True)

        episode_rewards.append(episode_reward)
        episode_steps.append(episode_step)

        average_reward = sum(episode_rewards) / len(episode_rewards)
        average_steps = sum(episode_steps) / len(episode_steps)

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

train_agent(agent_qlearn, maze, num_episodes=100)