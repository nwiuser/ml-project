import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import time
import random
from qlearn import QLearningAgent, actions
from maze import Maze, maze_fire, maze_medium, maze_piece_cake


# <------ les paramétres -------->
    
goal_reward = 100
wall_penalty = -10
step_penalty = 10

# maze_piece_cake.goal_position = (9,9)
# maze_medium.goal_position = (20,20)
# maze_fire.goal_position = (43,35)

maze = Maze(maze_fire, (0,0), (43,35))
agent = QLearningAgent(maze=maze)

# <------- les fonctions ------->

def finish_episode(agent, maze, current_episode, train=True):
    current_state = maze.start_position
    done = False
    episode_reward = 0
    episode_step = 0
    path = [current_state]

    while not done:
        # Choisir l'action basée sur l'état actuel
        action = agent.choose_action(current_state, current_episode)
        state_1 = current_state[0] + actions[action][0]
        state_2 = current_state[1] + actions[action][1]
        next_state = (state_1, state_2)

        # Vérification des limites et des murs
        if (
            next_state[0] < 0 or next_state[0] >= maze.width or
            next_state[1] < 0 or next_state[1] >= maze.height
        ):
            # Pénalité pour une tentative hors limites
            reward = wall_penalty
            next_state = current_state

        elif maze.maze[next_state[1]][next_state[0]] == 1:
            # Pénalité pour heurter un mur
            reward = wall_penalty
            next_state = current_state

        elif next_state == maze.goal_position:
            # Récompense pour atteindre l'objectif
            path.append(next_state)
            reward = goal_reward
            done = True

        else:
            # Mouvement valide
            path.append(next_state)
            reward = step_penalty

        # Mise à jour des récompenses et des étapes
        episode_reward += reward
        episode_step += 1

        # Entraînement si activé
        if train:
            agent.update_q_table(current_state, action, reward, next_state)

        # Mise à jour de l'état courant
        current_state = next_state

    return episode_reward, episode_step, path



def test_agent(agent, maze, num_episodes=1):
  episode_reward, episode_step, path = finish_episode(agent, maze, 1, train=False)

  print("Learned Path:")
  for row, col in path:
    print(f"({row}, {col})-> ", end='')
  print('Goal!')

  print("Number of steps : ", episode_step)
  print("Total reward : ", episode_reward)

  # Clear the existing plot if any
  if plt.gcf().get_axes():
    plt.cla()

  # Visuualize the maze using matplotlib
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

  # Mark the start position (red, 'S') and goal position (green 'G') in the maze
  plt.text(maze.start_position[0], maze.start_position[1], 'S', ha='center', va='center', color='red', fontsize=fontsize)
  plt.text(maze.goal_position[0], maze.goal_position[1], 'G', ha='center', va='center', color='green', fontsize=fontsize)

  # Mark the agent's path with blue '#' symbols
  for position in path:
    plt.text(position[0], position[1], '#', va='center', color='yellow', fontsize=fontsize)

  # Remove axis ticks and grid lines for a cleaner visualization
  plt.xticks([]), plt.yticks([])
  plt.grid(color='black', linewidth=2)
  plt.show()

  return episode_reward, episode_step, path



def train_agent(agent, maze, num_episodes=100):
    st.title("Entraînement de l'Agent - Visualisation en Temps Réel")
    
    # Variables pour stocker les récompenses et les étapes par épisode
    episode_rewards = []
    episode_steps = []

    # Placeholders Streamlit pour mettre à jour les graphiques et statistiques
    reward_chart = st.empty()
    steps_chart = st.empty()
    stats = st.empty()
    progress_bar = st.progress(0)

    # Démarrer le chronomètre
    start_time = time.time()

    for episode in range(num_episodes):
        # Terminer un épisode et collecter les résultats
        episode_reward, episode_step, path = finish_episode(agent, maze, episode, train=True)

        episode_rewards.append(episode_reward)
        episode_steps.append(episode_step)

        # Afficher les statistiques moyennes après chaque épisode
        average_reward = sum(episode_rewards) / len(episode_rewards)
        average_steps = sum(episode_steps) / len(episode_steps)

        elapsed_time = time.time() - start_time  # Temps écoulé en secondes
        formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))  # Format en heures:minutes:secondes

        stats.markdown(f"""
        **Épisode : {episode + 1}/{num_episodes}**
        - Récompense cumulée : {episode_reward}
        - Pas pris : {episode_step}
        - Récompense moyenne : {average_reward:.2f}
        - Nombre moyen de pas : {average_steps:.2f}
        - Temps écoulé : {formatted_time}
        """)

        # Graphique des récompenses
        with reward_chart.container():
            fig, ax = plt.subplots()
            ax.plot(episode_rewards, label="Récompense par épisode", color="blue")
            ax.set_title("Récompenses cumulées par épisode")
            ax.set_xlabel("Épisode")
            ax.set_ylabel("Récompense cumulée")
            ax.legend()
            st.pyplot(fig)

        # Graphique des étapes par épisode
        with steps_chart.container():
            fig, ax = plt.subplots()
            ax.plot(episode_steps, label="Étapes par épisode", color="green")
            ax.set_title("Nombre d'étapes par épisode")
            ax.set_xlabel("Épisode")
            ax.set_ylabel("Étapes prises")
            ax.legend()
            st.pyplot(fig)

        # Mettre à jour la barre de progression
        progress_bar.progress((episode + 1) / num_episodes)

    # Calcul du temps total d'entraînement
    total_time = time.time() - start_time
    formatted_total_time = time.strftime("%H:%M:%S", time.gmtime(total_time))

    # Résumé global après la fin de l'entraînement
    st.success("Entraînement terminé ! 🎉")
    st.write(f"Récompense moyenne : {average_reward:.2f}")
    st.write(f"Nombre moyen de pas : {average_steps:.2f}")
    st.write(f"Temps total d'entraînement : {formatted_total_time}")



# Training and testing
# test_agent(agent, maze, num_episodes=100)
train_agent(agent, maze, num_episodes=100)
# test_agent(agent, maze, num_episodes=100)