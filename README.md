# Flappy Bird AI using Deep Q-Learning

A Reinforcement Learning project that trains a Deep Q-Network (DQN) agent to play Flappy Bird using PyTorch and Gymnasium.

The agent learns to choose actions by interacting with the environment, receiving rewards, storing experiences in replay memory, and updating its neural network using Deep Q-Learning.

## Overview

The objective of this project is to build an AI agent capable of learning how to play Flappy Bird without explicitly programming the gameplay strategy.

At each step, the agent:

1. Observes the current game state.
2. Selects an action.
3. Receives a reward from the environment.
4. Observes the next state.
5. Stores the experience in replay memory.
6. Learns from a randomly sampled mini-batch.

## Technologies Used

* Python
* PyTorch
* Gymnasium
* Flappy Bird Gymnasium
* PyYAML

## Reinforcement Learning Approach

This project uses a **Deep Q-Network (DQN)**.

The DQN estimates the Q-value of each possible action for a given state. The agent then selects the action with the highest predicted Q-value during exploitation.

The project uses two neural networks:

### Policy Network

The policy network predicts the Q-values for the available actions and is updated during training.

### Target Network

The target network is used to calculate stable target Q-values. It is periodically synchronized with the policy network to improve training stability.

## Exploration and Exploitation

The agent uses an **epsilon-greedy strategy**.

During training:

* With probability `epsilon`, the agent chooses a random action.
* Otherwise, it chooses the action with the highest predicted Q-value.

The value of epsilon gradually decreases during training, allowing the agent to transition from exploration to exploitation.

## Experience Replay

The agent stores its experiences in replay memory in the form:

```text
(state, action, next_state, reward, terminated)
```

During training, a random mini-batch is sampled from this memory.

This helps reduce the correlation between consecutive experiences and makes DQN training more stable.

## Project Structure

```text
flappy_bird/
│
├── agent.py
├── dqn.py
├── experience_replay.py
├── parameters.yaml
└── README.md
```

### File Description

| File                   | Description                         |
| ---------------------- | ----------------------------------- |
| `agent.py`             | Main training and evaluation logic  |
| `dqn.py`               | Defines the Deep Q-Network          |
| `experience_replay.py` | Implements experience replay memory |
| `parameters.yaml`      | Contains the DQN hyperparameters    |
| `README.md`            | Project documentation               |

## Hyperparameters

The training parameters are stored in `parameters.yaml`.

These include:

* Learning rate
* Discount factor (`gamma`)
* Initial epsilon
* Minimum epsilon
* Epsilon decay
* Replay memory size
* Mini-batch size
* Target network synchronization rate
* Reward threshold

Keeping the parameters in a separate YAML file makes it easier to experiment with different configurations.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/flappy-bird-dqn.git
cd flappy-bird-dqn
```

Install the required dependencies:

```bash
pip install torch gymnasium flappy-bird-gymnasium pyyaml
```

## Training

Run the following command to train the DQN agent:

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

The trained model is saved as:

```text
runs/flappybirdv0.pt
```

## Evaluation

To run the trained agent and watch it play Flappy Bird:

```bash
python agent.py flappybirdv0
```

The game will open in a rendered window and the trained agent will play automatically.

To stop the program:

```text
Ctrl + C
```

## Results

After training, the agent was able to learn a functional policy for navigating through the pipes.

During evaluation, the trained model achieved rewards such as:

```text
Episode 3  → 12.9
Episode 9  → 17.4
Episode 12 → 18.7
Episode 13 → 21.0
Episode 18 → 18.7
```

These results show that the agent successfully learned to interact with the environment and survive for longer periods compared with its initial training performance.

## How DQN Works in This Project

```text
        Game Environment
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
        ┌───────┴────────┐
        ▼                ▼
     Reward         Next State
        │                │
        └───────┬────────┘
                ▼
        Experience Replay
                │
                ▼
          Sample Mini-batch
                │
                ▼
        Calculate Target Q
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

This project demonstrates practical implementation of:

* Reinforcement Learning
* Q-Learning
* Deep Q-Networks
* Epsilon-greedy exploration
* Experience replay
* Target networks
* Bellman equation
* Q-value estimation
* Neural network optimization
* PyTorch model training

## Future Improvements

Possible improvements include:

* Hyperparameter optimization
* Reward visualization
* Training performance graphs
* Improved checkpoint management
* Double DQN
* Dueling DQN
* More stable training and evaluation
* Automatic performance monitoring

## Purpose

This project was built as a practical implementation of Deep Reinforcement Learning to understand how an agent can learn a task through interaction with an environment rather than being explicitly programmed with a fixed strategy.
