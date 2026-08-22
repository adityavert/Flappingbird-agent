# Flappy Bird AI — Deep Q-Learning

A Flappy Bird project featuring both a **normal playable version of the game** and an **AI agent trained using Deep Q-Learning (DQN)**.

The AI learns to play Flappy Bird through interaction with the game environment using PyTorch, experience replay, a target network, and epsilon-greedy exploration.

## Overview

This project has two main components:

1. **Normal Flappy Bird** — a playable version of the game where the player controls the bird.
2. **Flappy Bird AI** — a Reinforcement Learning agent that learns to play the game using Deep Q-Learning.

The project was built to explore how Reinforcement Learning can be applied to a simple game environment and to understand the practical implementation of DQN.

## Features

* Playable Flappy Bird game
* AI-controlled Flappy Bird
* Deep Q-Network (DQN)
* Experience replay
* Policy and target networks
* Epsilon-greedy exploration
* Configurable hyperparameters
* PyTorch-based training
* Model saving and evaluation

## Technologies Used

* Python
* PyTorch
* Gymnasium
* Flappy Bird Gymnasium
* PyYAML

## Project Structure

```text
flappy_bird/
│
├── game_flappy_bird.py
├── agent.py
├── dqn.py
├── experience_replay.py
├── parameters.yaml
└── README.md
```

### File Description

| File                   | Description                            |
| ---------------------- | -------------------------------------- |
| `game_flappy_bird.py`       | Normal playable Flappy Bird game       |
| `agent.py`             | Main DQN training and evaluation logic |
| `dqn.py`               | Defines the Deep Q-Network             |
| `experience_replay.py` | Implements experience replay memory    |
| `parameters.yaml`      | Contains the training hyperparameters  |
| `README.md`            | Project documentation                  |

## Normal Flappy Bird

The project includes a normal playable version of Flappy Bird.

Run the game using:

```bash
python game_flappy_bird.py
```

The player controls the bird and attempts to navigate through the pipes while achieving the highest possible score.

## AI Flappy Bird

The AI version uses a **Deep Q-Network (DQN)** to learn how to play Flappy Bird.

Instead of manually programming the bird's behavior, the agent learns by repeatedly interacting with the environment.

At every step, the agent:

1. Observes the current state.
2. Selects an action.
3. Receives a reward.
4. Observes the next state.
5. Stores the experience.
6. Learns from previously collected experiences.

## Deep Q-Network

The DQN estimates the Q-value of each possible action for a given state.

The project uses two neural networks:

### Policy Network

The policy network predicts Q-values for the available actions and is updated during training.

### Target Network

The target network is used to calculate stable target Q-values and is periodically synchronized with the policy network.

This helps make the training process more stable.

## Exploration vs Exploitation

The agent uses an **epsilon-greedy strategy**.

During training:

* With probability `epsilon`, the agent selects a random action.
* Otherwise, it selects the action with the highest predicted Q-value.

The epsilon value gradually decreases during training, allowing the agent to explore the environment initially and increasingly rely on its learned policy.

## Experience Replay

The agent stores experiences in replay memory in the form:

```text
(state, action, next_state, reward, terminated)
```

A random mini-batch is sampled from the replay memory during training.

This reduces the correlation between consecutive experiences and improves the stability of DQN training.

## Hyperparameters

The DQN training parameters are stored in:

```text
parameters.yaml
```

The configuration includes:

* Learning rate (`alpha`)
* Discount factor (`gamma`)
* Initial epsilon
* Minimum epsilon
* Epsilon decay
* Replay memory size
* Mini-batch size
* Target network synchronization rate
* Reward threshold

Keeping these values separate from the Python code makes it easier to experiment with different configurations.

## Installation

Clone the repository:

```bash
https://github.com/adityavert/Flappybird-agent.git
cd Flappybird-agent
```

Install the required dependencies:

```bash
pip install torch gymnasium flappy-bird-gymnasium pyyaml
```

## Training the AI

To train the DQN agent:

```bash
python agent.py flappybirdv0 --train
```

During training, the terminal displays the episode number, total reward, and epsilon value.

Example:

```text
for episode=1,with total rewards=...
for episode=2,with total rewards=...
for episode=3,with total rewards=...
```

The trained model is saved in:

```text
runs/flappybirdv0.pt
```

## Running the AI

To evaluate the trained model:

```bash
python agent.py flappybirdv0
```

The Flappy Bird environment will be rendered and the trained agent will play automatically.

To stop the evaluation:

```text
Ctrl + C
```

## Results

After training, the DQN agent learned a functional policy for navigating through the pipes.

During evaluation, the trained agent consistently achieved positive rewards, with observed episode rewards reaching approximately 21 during testing.

```text
Episode 3  → 12.9
Episode 9  → 17.4
Episode 12 → 18.7
Episode 13 → 21.0
Episode 18 → 18.7
```

The results demonstrate that the agent was able to learn useful behavior through interaction with the environment.

## DQN Training Process

```text
             Flappy Bird Environment
                       │
                       ▼
                 Current State
                       │
                       ▼
                  Policy DQN
                       │
                       ▼
                  Select Action
                       │
                       ▼
              Environment Step
                       │
                 ┌─────┴─────┐
                 ▼           ▼
              Reward     Next State
                 │           │
                 └─────┬─────┘
                       ▼
               Experience Replay
                       │
                       ▼
                 Sample Batch
                       │
                       ▼
                Calculate Target
                       │
                       ▼
                Update Policy DQN
                       │
                       ▼
               Sync Target Network
                       │
                       ▼
                    Repeat
```

## Learning Outcomes

This project provides practical experience with:

* Reinforcement Learning
* Q-Learning
* Deep Q-Networks
* Epsilon-greedy exploration
* Experience replay
* Target networks
* Bellman equation
* Q-value estimation
* Neural network optimization
* PyTorch
* Gymnasium environments

## Future Improvements

* Hyperparameter optimization
* Reward and loss visualization
* Training performance graphs
* Double DQN
* Dueling DQN
* Improved training stability
* Automated evaluation
* Comparison of different DQN architectures

## Purpose

This project was developed as a practical exploration of **Deep Reinforcement Learning**, demonstrating how an AI agent can learn to play a game through interaction with its environment instead of relying on a manually programmed strategy.
