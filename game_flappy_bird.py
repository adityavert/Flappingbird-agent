import gymnasium as gym
import flappy_bird_gymnasium
import pygame

# Creating our env
env = gym.make("FlappyBird-v0", render_mode="human")
state, info = env.reset()
done = False

# Initialize PyGame keyboard
pygame.init()
screen = pygame.display.get_surface()  # Gym has already created a window

while not done:
    action = 0  # 0 = no flap, 1 = flap

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                action = 1  # flap

    state, reward, done, truncated, info = env.step(action)
    env.render()

env.close()
pygame.quit()