import streamlit as st
from maze import maze_medium, maze_piece_cake
from qlearn import QLearningAgent
from montecarlo import MonteCarloAgent
# from montecarlo import MonteCarloAgent
from ql_train import train_agent
from maze import Maze
# from mc_train import train_monte_carlo_agent

maze = Maze(maze_medium)

# Titre principal
st.title("Teston ton agent")

# # Section 1 : Choisir un labyrinthe
# st.subheader("Choisir un agent")
# col1, col2 = st.columns(2)

# with col1:
#     if st.button("Piece Cake"):
#         maze = maze_piece_cake
#         st.write(maze)

# with col2:
#     if st.button("Medium"):
#         maze = maze_medium
#         st.write(maze)
        

# Section 2 : Choisir un agent
st.subheader("Choisir un agent")
col3, col4 = st.columns(2)

with col3:
    if st.button("Q-Learning"):
        agent_ql = QLearningAgent(maze)

with col4:
    if st.button("Monte Carlo Tree Selection"):
        agent_mc = MonteCarloAgent(maze)
        
        
st.subheader("Ajuster les hyperparamètres")

goal_reward = st.slider("Goal Reward", min_value=-10.0, max_value=10.0, value=1.0, step=0.1)
wall_penalty = st.slider("Wall Penalty", min_value=-10.0, max_value=10.0, value=-1.0, step=0.1)
step_reward = st.slider("Step Reward", min_value=-10.0, max_value=10.0, value=-0.1, step=0.1)

st.write("Hyperparamètres choisis :")
st.write(f"- Goal Reward : {goal_reward}")
st.write(f"- Wall Penalty : {wall_penalty}")
st.write(f"- Step Reward : {step_reward}")

if st.button("Lancer l'entraînement"):
    if agent_ql:
        train_agent(agent_ql, maze, goal_reward, wall_penalty, step_reward)