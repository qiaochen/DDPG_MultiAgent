from unityagents import UnityEnvironment
import numpy as np
from agent import MADDPG
from utils import draw

unity_environment_path = "./Tennis_Linux/Tennis.x86_64"
best_model_path = "./best_model.checkpoint"
rollout_length = 3

if __name__ == "__main__":
    # prepare environment
    env = UnityEnvironment(file_name=unity_environment_path)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space 
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])

    num_episodes = 2500
    agent = MADDPG(state_size, 
                   action_size,
                   lr_actor = 1e-5,
                   lr_critic = 1e-4,
                   lr_decay = .995,
                   replay_buff_size = int(1e6),
                   gamma = .95,
                   batch_size = 64,
                   random_seed = 999,
                   soft_update_tau = 1e-3
                 )
    
    total_rewards = []
    avg_scores = []
    max_avg_score = -1
    max_score = -1
    threshold_init = 20
    noise_t = 1.0
    noise_decay = .995
    worsen_tolerance = threshold_init  # for early-stopping training if consistently worsen for # episodes
    for i_episode in range(1, num_episodes+1):
        env_inst = env.reset(train_mode=True)[brain_name]    # reset the environment
        states = env_inst.vector_observations                # get the current state
        scores = np.zeros(num_agents)                        # initialize score array
        dones = [False]*num_agents
        while not np.any(dones):
            actions = agent.act(states,noise_t)              # select an action
            env_inst = env.step(actions)[brain_name]         # send the action to the environment
            next_states = env_inst.vector_observations       # get the next state
            rewards = env_inst.rewards                       # get the reward
            dones = env_inst.local_done                      # see if episode has finished
            agent.update(states, actions, rewards, next_states, dones)
            
            noise_t *= noise_decay
            scores += rewards                                # update scores
            states = next_states 
        
        episode_score = np.max(scores)
        total_rewards.append(episode_score)
        print("Episodic {} Score: {:.4f}".format(i_episode, episode_score))
        
        if max_score <= episode_score:                     
            max_score = episode_score
            agent.save(best_model_path)                     # save best model so far
        
        if len(total_rewards) >= 100:                       # record avg score for the latest 100 steps
            latest_avg_score = sum(total_rewards[(len(total_rewards)-100):]) / 100
            print("100 Episodic Everage Score: {:.4f}".format(latest_avg_score))
            avg_scores.append(latest_avg_score)
          
            if max_avg_score <= latest_avg_score:           # record better results
                worsen_tolerance = threshold_init           # re-count tolerance
                max_avg_score = latest_avg_score
            else:                                           
                if max_avg_score > 0.5:                     
                    worsen_tolerance -= 1                   # count worsening counts
                    print("Loaded from last best model.")
                    agent.load(best_model_path)             # continue from last best-model
                if worsen_tolerance <= 0:                   # earliy stop training
                    print("Early Stop Training.")
                    break
                    
    draw(total_rewards,"./training_score_plot.png", "Training Scores (Per Episode)")
    draw(avg_scores,"./training_100avgscore_plot.png", "Training Scores (Average of Latest 100 Episodes)", ylabel="Avg. Score")
    env.close()

                    