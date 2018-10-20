# Project Report

In this project, a multi-agent for tennis player control is trained using Deep Deterministic Policy Gradient (DDPG). Through learning by self-playing, the agent obtained good performance in the evaluation test.


## Learning Algorithm

- Network Architecture
  The architecture of the Actor and Critic networks are summarized using the project [pytorch-summary](https://github.com/sksq96/pytorch-summary/) as follows:
  - Actor Network
  ```
  ----------------------------------------------------------------
          Layer (type)               Output Shape         Param #
  ================================================================
              Linear-1                [-1, 1, 64]           1,600
              Linear-2               [-1, 1, 128]           8,320
              Linear-3                 [-1, 1, 2]             258
  ================================================================
  Total params: 10,178
  Trainable params: 10,178
  Non-trainable params: 0
  ----------------------------------------------------------------
  Input size (MB): 0.00
  Forward/backward pass size (MB): 0.00
  Params size (MB): 0.04
  Estimated Total Size (MB): 0.04
  ----------------------------------------------------------------
  ```
  - Critic Network
  ```
  ----------------------------------------------------------------
          Layer (type)               Output Shape         Param #
  ================================================================
              Linear-1                   [-1, 64]           1,600
              Linear-2                  [-1, 128]           8,576
              Linear-3                   [-1, 64]           8,256
              Linear-4                    [-1, 1]              65
  ================================================================
  Total params: 18,497
  Trainable params: 18,497
  Non-trainable params: 0
  ----------------------------------------------------------------
  Input size (MB): 0.00
  Forward/backward pass size (MB): 0.00
  Params size (MB): 0.07
  Estimated Total Size (MB): 0.07
  ----------------------------------------------------------------
  ```
- Hyper-parameters
  - learning rate for actor network: 1e-5
  - learning rate for the critic network: 1e-4
  - learning rate decay rate: 0.995
  - replay buffer size: 1e6
  - long term reward discount rate: 0.95
  - soft update tau: 0.001
- Training Strategy
  - Adam is used as the optimizer
  - An `early-stop` scheme is applied to stop training if the 100-episode-average score continues decreasing over `20` consecutive episodes.
  - Each time the model gets worse regarding avg scores, the model recovers from the last best model and the learning rate of Adam is decreased: `new learning rate = old learning rate * learning rate decay rate` 

## Performance Evaluation
### Training
During training, the performance jumped to the best level and stabalized there after about **1700** episodes. Before that, the first time the performance surpassed 0.5 occurred at around episode 800. The episodic and average (over 100 latest episodes) scores are plotted as following:
- Reward per-episode during training

![img](https://raw.githubusercontent.com/qiaochen/DDPG_MultiAgent/master/training_score_plot.png)

- Average reward over latest 100 episodes during training

![img](https://raw.githubusercontent.com/qiaochen/DDPG_MultiAgent/master/training_100avgscore_plot.png)

As can be seen from the plot, the average score gradually passed **0.5** and reached **2.0** during training, before the early-stopping scheme terminates the training process.

### Testing
The scores of 100 testing episodes are visualized as follows:
![img](https://raw.githubusercontent.com/qiaochen/DDPG_MultiAgent/master/test_score_plot.png)
The model obtained an average score of **+1.95** during testing, which is over **+0.5**.

## Conclusion
The trained model has successfully solved the tennis play task. The performance:
1. an average score of `+1.95` over `100` consecutive episodes 
2. the best model was trained using around `1700` episodes

has fulfilled the passing threshold of solving the problem: obtain an average score of higher than `+0.5` over `100` consecutive episodes.

## Ideas for Future Work
- Use prioritized replay buffer, or Rainbow to improve the Critic network
- Use methods like GAE or PPO in the calculation of policy loss, to improve the training performance of the Actor network.
- See if A2C and other algorithms could perform better.
