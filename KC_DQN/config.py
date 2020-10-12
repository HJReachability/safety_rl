# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

class config():
    def __init__(self,  ENV_NAME='Pendulum-v0',
                        DEVICE='cpu',
                        MAX_EPISODES=200, MAX_EP_STEPS=200,
                        LR_C=1e-3, LR_C_END=1e-4,
                        EPSILON=0.9, EPSILON_END=5e-2,
                        GAMMA=0.9, 
                        TAU=0.01, HARD_UPDATE=200, SOFT_UPDATE=True,
                        MEMORY_CAPACITY=10000,
                        BATCH_SIZE=32,
                        RENDER=False,                         
                        MAX_MODEL=5):
                
        self.MAX_EPISODES = MAX_EPISODES
        self.MAX_EP_STEPS = MAX_EP_STEPS
        
        self.LR_C = LR_C
        self.LR_C_END = LR_C_END
        self.EPSILON = EPSILON
        self.EPSILON_END = EPSILON_END
        
        self.MEMORY_CAPACITY = MEMORY_CAPACITY
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        
        self.TAU = TAU
        self.HARD_UPDATE = HARD_UPDATE
        self.SOFT_UPDATE = SOFT_UPDATE
        
        self.RENDER = RENDER
        self.ENV_NAME = ENV_NAME
        
        self.MAX_MODEL = MAX_MODEL
        self.DEVICE=DEVICE
        
class dqnConfig(config):
    def __init__(self,  ENV_NAME='Pendulum-v0',
                        DEVICE='cpu',
                        MAX_EPISODES=200, MAX_EP_STEPS=200,
                        LR_C=1e-3, LR_C_END=1e-4,
                        EPSILON=0.9, EPSILON_END=5e-2,
                        GAMMA=0.9, 
                        TAU=0.01, HARD_UPDATE=200, SOFT_UPDATE=True,
                        MEMORY_CAPACITY=10000,
                        BATCH_SIZE=32,
                        RENDER=False,                         
                        MAX_MODEL=5,
                        DOUBLE=True):
        
        super().__init__(ENV_NAME=ENV_NAME,
                        DEVICE=DEVICE,
                        MAX_EPISODES=MAX_EPISODES, MAX_EP_STEPS=MAX_EP_STEPS,
                        LR_C=LR_C, LR_C_END=LR_C_END,
                        EPSILON=EPSILON, EPSILON_END=EPSILON_END,
                        GAMMA=GAMMA, 
                        TAU=TAU, HARD_UPDATE=HARD_UPDATE, SOFT_UPDATE=SOFT_UPDATE,
                        MEMORY_CAPACITY=MEMORY_CAPACITY,
                        BATCH_SIZE=BATCH_SIZE,
                        RENDER=RENDER,                         
                        MAX_MODEL=MAX_MODEL)
        self.DOUBLE = DOUBLE

#== for actor-critic, DDPG
# self.LR_A = LR_A
# self.LR_A_END = LR_A_END
#== for DDPG
# self.SIGMA = SIGMA