# Project Details
---
In this project, two Actor-Critic agents were trained using Deep Deterministic Policy Gradients (DDPG) to play the tennis game:

![](https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif)

## The Environment
The environment for this project involves two agents controlling rackets to bounce a ball over a net.
### State Space
State is continuous, it has **8** dimensions corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.
### Action Space
Each action is continuous, in the form of a vector with **2** dimensions, corresponding to movement toward (or away from) the net, and jumping.
### Reward
- A reward of **+0.1** is obtained, if an agent hits the ball over the net. 
- A reward of **-0.01** is obtained, if an agent lets a ball hit the ground or hits the ball out of bounds.
### Goal
The agents must bounce ball between one another while not dropping or sending ball out of bounds. The longer the bounce turns last, the better the performance is achieved.
### Solving the Environment
The task is episodic. An average score of **+0.5** (over **100** consecutive episodes, after taking the maximum of the two agents) is required to solve this task.

# Getting Started
## Step 1: Clone the Project and Install Dependencies
\*Please prepare a python3 virtual environment if necessary.
```
git clone https://github.com/qiaochen/DDPG_MultiAgent
cd install_requirements
pip install .
```
## Step 2: Download the Unity Environment
For this project, I use the environment form **Udacity**. The links to modules at different system environments are copied here for convenience:
*   Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
*   Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
*   Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
*   Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

I conducted my experiments in Ubuntu 16.04, so I picked the 1st option. Then, extract and place the Tennis_Linux folder within the project root. The project folder structure now looks like this (Project program generated .png and model files are excluded):
```
Project Root
     |-install_requirements (Folder)
     |-README.md
     |-Report.md
     |-agent.py
     |-models.py
     |-train.py
     |-test.py
     |-utils.py
     |-Tennis_Linux (Folder)
            |-Tennis.x86_64
            |-Tennis.x86
            |-Tennis_Data (Folder)
```
## Instructions to the Program
---
### Step 1: Training
```
python thain.py
```
After training, the following files will be generated and placed in the project root folder:

- best_model.checkpoint (the trained model)
- training_100avgscore_plot.png (a plot of avg. scores during training)
- training_score_plot.png (a plot of per-episode scores during training)
- unity-environment.log (log file created by Unity)
### Step 2: Testing
```
python test.py
```
The testing performance will be summarized in the generated plot within project root:

- test_score_plot.png
