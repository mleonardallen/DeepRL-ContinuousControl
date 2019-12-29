from collections import deque
import numpy as np
import matplotlib.pyplot as plt

def episode(agent):

  config = agent.config
  env = config.get('env')
  train_mode = config.get('train_mode', True)
  brain_name = env.brain_names[0]
  env_info = env.reset(train_mode)[brain_name]
  num_agents = len(env_info.agents)
  states = env_info.vector_observations
  scores = np.zeros(num_agents)

  while True:
    actions_info = agent.act(states)

    if train_mode:
      actions = actions_info['action'].detach().numpy() # extract actions from dict
      actions = np.clip(actions, -1, 1) # possible for sampled actions to be outside (-1, 1)
    else:
      actions = actions_info['mean'].detach().numpy()

    env_info = env.step(actions)[brain_name]
    next_states = env_info.vector_observations
    rewards = env_info.rewards
    dones = env_info.local_done
    scores += env_info.rewards

    if train_mode:
      agent.step(states, actions_info, rewards, next_states, dones)

    states = next_states

    if np.any(dones):
      break
  
  return np.mean(scores)

def run(agent):

  config = agent.config
  episodes = config.get('episodes', 1)
  print_every = config.get('print_every', None)
  plot_every = config.get('plot_every', None)

  plot_scores = []
  tmp_scores = deque(maxlen=100)

  for i in range(1, episodes+1):
    score = episode(agent)
    tmp_scores.append(score)
    if print_every and i % print_every == 0:
      print('Episode: {}, Total score (averaged over agents): {}'.format(i, score))
    if plot_every and i % plot_every == 0:
      plot_scores.append((i, np.mean(tmp_scores)))

  if len(plot_scores):
    x, y = zip(*plot_scores)
    plt.plot(x, y)
    plt.xlabel('Episode')
    plt.ylabel('Average Score')
    plt.show()
