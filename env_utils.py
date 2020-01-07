from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import progressbar as pb

def get_rolling_avg(values, n):
  tmp_values = deque(maxlen=n)
  avg_values = []
  for value in values:
    tmp_values.append(value)
    avg_values.append(np.mean(tmp_values))
  return avg_values

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
    actions = actions_info['actions']
    actions = actions.detach().numpy()
    actions = np.clip(actions, -1, 1) # samples can fall outside (-1, 1)
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

  train_mode = config.get('train_mode', True)
  if train_mode:
    timer = pb.ProgressBar(
      widgets=[
        'Episode: ', 
        pb.SimpleProgress(), ' ',
        pb.Variable('Score'), ' ',
        pb.AdaptiveETA()
      ],
      maxval=episodes
    ).start()

  scores = []
  for i in range(1, episodes+1):
    score = episode(agent)
    scores.append(score)
    if train_mode:
      timer.update(i, Score=score)
  if train_mode:
    timer.finish()
  return scores

def plot_episodes(values, label, axis=None, avg_n=100):

  if axis == None:
    axis = plt

  avg_values = get_rolling_avg(values, avg_n)

  x = range(len(values))
  axis.plot(x, values, label=label, color='lightblue')
  axis.plot(x, avg_values, label="Average over {} episodes".format(avg_n), color='blue')
  axis.legend()
  
  if hasattr(axis, 'set_xlabel'):
    axis.set_xlabel('Episode')
    axis.set_ylabel(label)
  else:
    axis.xlabel('Episode')
    axis.ylabel(label)
  
  if hasattr(axis, 'show'):
    axis.show()
