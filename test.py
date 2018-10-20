from unityagents import UnityEnvironment
import numpy as np
from tqdm import tqdm
from agent import MADDPG
from utils import draw

unity_environment_path = "./Tennis_Linux/Tennis.x86_64"
best_model_path = "./best_model.checkpoint"

if __name__ == "__main__":
    # prepare environment
    env = UnityEnvironment(file_name=unity_environment_path)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]

    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # dim of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)
    
    # dim of the state space 
    states = env_info.vector_observations
    state_size = states.shape[1]
    
    agent = MADDPG(state_size, action_size)
                    
    agent.load(best_model_path)

    test_scores = []
    for i_episode in tqdm(range(1, 101)):
        scores = np.zeros(num_agents)                       # initialize the scores
        env_info = env.reset(train_mode=True)[brain_name]   # reset the environment
        states = env_info.vector_observations               # get the current states
        dones = [False]*num_agents
        while not np.any(dones):
            actions = agent.act(states)                     # select actions
            env_info = env.step(actions)[brain_name]        # send the actions to the environment
            next_states = env_info.vector_observations      # get the next states
            rewards = env_info.rewards                      # get the rewards
            dones = env_info.local_done                     # see if episode has finished
            scores += rewards                               # update the scores
            states = next_states                            # roll over the states to next time step

        test_scores.append(np.max(scores))
                
    avg_score = sum(test_scores)/len(test_scores)
    print("Test Score: {}".format(avg_score))
    draw(test_scores, "./test_score_plot.png", "Test Scores of 100 Episodes (Avg. score {})".format(avg_score))
    env.close()
