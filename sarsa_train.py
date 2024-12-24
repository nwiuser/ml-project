import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import time
from fonctions import finish_episode  
from maze import Maze
from params import agent_sarsa, maze

actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def train_sarsa_agent(agent, maze, num_episodes=100):
    st.title("Entraînement de l'Agent SARSA - Visualisation en Temps Réel")

  
    episode_rewards = []
    episode_steps = []


    reward_chart = st.empty()
    steps_chart = st.empty()
    stats = st.empty()
    progress_bar = st.progress(0)

   
    start_time = time.time()

    for episode in range(num_episodes):
       
        episode_reward, episode_step, path, states, rewards, actions = finish_episode(agent, maze, episode, train=True)

       
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

train_sarsa_agent(agent_sarsa, maze, num_episodes=100)
