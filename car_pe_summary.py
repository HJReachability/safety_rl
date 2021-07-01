#!/usr/bin/env python
# coding: utf-8

# In[1]:


from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning)

from gym_reachability import gym_reachability  # Custom Gym env.
import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
import torch
import pickle
import os

from KC_DQN.DDQNPursuitEvasion import DDQNPursuitEvasion
from KC_DQN.config import dqnConfig

import time
timestr = time.strftime("%Y-%m-%d-%H_%M")


# In[2]:


def getModelInfo(path):
    dataFolder = os.path.join('scratch', path)
    modelFolder = os.path.join(dataFolder, 'model')
    picklePath = os.path.join(modelFolder, 'CONFIG.pkl')
    with open(picklePath, 'rb') as fp:
        config = pickle.load(fp)
    config.DEVICE = 'cpu'

    dimList = [stateDim] + config.ARCHITECTURE + [actionNum]
    return dataFolder, config, dimList


# In[3]:


doneType='toEnd'
env_name = "dubins_car_pe-v0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make(env_name, device=device, mode='RA', doneType=doneType, sample_inside_obs=False, considerPursuerFailure=False)

stateDim = env.state.shape[0]
actionNum = env.action_space.n
action_list = np.arange(actionNum)

env.set_costParam(1, -1, 'sparse', 1)
env.set_seed(0)

env.report()


# In[4]:


dataFolder = os.path.join('scratch', 'car-pe-DDQN', '9999-fail')
estDict = np.load(os.path.join(dataFolder, 'data', 'estError.npy'), allow_pickle=True).tolist()
print(estDict.keys())


# In[5]:


ddqnValue = estDict['ddqnValue']
rolloutValue = estDict['rolloutValue']


# In[6]:


# False Success: ddqn <= 0, rollout > 0
FS_indices = np.logical_and(ddqnValue<=0, rolloutValue>0)
PS_indices = ddqnValue<=0
FS_num = np.sum(FS_indices)
PS_num = np.sum(PS_indices)

# False Failure: ddqn > 0, rollout <= 0
FF_indices = np.logical_and(ddqnValue>0, rolloutValue<=0)
PF_indices = ddqnValue>0
FF_num = np.sum(FF_indices)
PF_num = np.sum(PF_indices)

print(FS_num / PS_num)
print(FF_num / PF_num)

print( (FS_num + FF_num) / (PS_num + PF_num))

# print(np.sum(PS_indices) + np.sum(PF_indices))


# ## DEFENDER VALIDATION --- NEG

# In[7]:


from utils.carPEAnalysis import analyzeValidationResult, checkCapture, checkCrossConstraint


# In[8]:


validationFile = os.path.join(dataFolder, 'data', 'valDictNEG.npy')
valDictNeg, successList, failureList, captureList, captureInstantList,    crossConstraintList, crossConstraintInstantList, unfinishedList = analyzeValidationResult(validationFile, env, verbose=True)

samplesDict = np.load(os.path.join(dataFolder, 'data', 'NEG', 'samplesNEG.npy'), allow_pickle=True).tolist()


# In[9]:


dictList = valDictNeg['dictList']
exhaustiveValueList = valDictNeg['exhaustiveValueList']
states = valDictNeg['states']
stateIdxList = valDictNeg['stateIdxList']


# In[10]:


idx = 0
valDictTmp = np.load(os.path.join(dataFolder, 'data', 'NEG', 'valDictNEG'+str(idx)+'.npy'), allow_pickle=True).tolist()

print(dictList[idx]['maxminV'])
print(valDictTmp['dict']['maxminV'])
print(np.max(exhaustiveValueList[idx]))


# In[11]:


ddqnList = samplesDict['ddqnList']
rolloutPolicyValueList = samplesDict['rolloutPolicyValueList']
exhaustiveValueList = np.empty(shape=(len(states)))
for i in range(len(states)):
    exhaustiveValueList[i] = dictList[i]['maxminV']


# In[12]:


numActionList = env.numActionList
dataFolder, CONFIG, dimList = getModelInfo(os.path.join('car-pe-DDQN', '9999-fail'))
agent2 = DDQNPursuitEvasion(CONFIG, numActionList, dimList=dimList, mode='RA', terminalType='g')
agent2.restore(1000000, dataFolder)


# In[13]:


agentList = [agent2]
axStyle = env.get_axes()
nx = 101
ny = 101
vmin = -1
vmax = 1
xs = np.linspace(env.bounds[0,0], env.bounds[0,1], nx)
ys = np.linspace(env.bounds[1,0], env.bounds[1,1], ny)

fsz=20
nRow = len(agentList)
nCol = 2
subfigSz = 4
figsize =  (nCol*subfigSz, nRow*subfigSz)

fig, axArray = plt.subplots(nRow, nCol, figsize=figsize, sharex=True, sharey=True)
pick_capture = False

for i, agent in enumerate(agentList):
#     axes = axArray[i]
    axes = axArray
    
    pick_idx = 0
    if pick_capture:
        test_idx = captureList[pick_idx]
        endInstant = captureInstantList[pick_idx]
        print("For test [{}], the pursuer caught the evader at the {} step.".format(test_idx, endInstant))
    else:
        test_idx = crossConstraintList[pick_idx]
        endInstant = crossConstraintInstantList[pick_idx]
        print("For test [{}], the evader collided with the obstacle at the {} step.".format(test_idx, endInstant))

    state = states[test_idx]
    trajEvader = dictList[test_idx]['trajEvader']
    trajPursuer = dictList[test_idx]['trajPursuer']
    exhaustiveValue = dictList[test_idx]['maxminV']

    #= Rollout
    ax = axes[0]
    _, minVs = env.plot_trajectories(agent.Q_network, states=[state],
            toEnd=False, ax=ax, lw=2, T=200, c=['#0abab5', 'k'])
    env.plot_target_failure_set(ax=ax, showCapture=False)
    
    if i == nRow-1:
        ax.set_xlabel('Policy Rollout: {:.3f}'.format(minVs[0]), fontsize=fsz)
        
    #= Exhaustive
    ax = axes[1]
    ax.plot(trajEvader[:endInstant, 0],  trajEvader[:endInstant, 1],  lw=2, c='#0abab5')
    ax.plot(trajPursuer[:endInstant, 0], trajPursuer[:endInstant, 1], lw=2, c='k')
    ax.scatter(trajEvader[0, 0],  trajEvader[0, 1],  s=48, c='#0abab5', zorder=3)
    ax.scatter(trajPursuer[0, 0], trajPursuer[0, 1], s=48, c='k', zorder=3)
    
    if pick_capture:
        xPursuer, yPursuer = trajPursuer[endInstant, :2]
        xEvader,  yEvader  = trajEvader[endInstant, :2]
        
        env.plot_target_failure_set(ax=ax, xPursuer=xPursuer, yPursuer=yPursuer)
        ax.scatter(xPursuer, yPursuer, s=48, c='b', marker='x', zorder=3)
        ax.scatter(xEvader,  yEvader,  s=48, c='b', marker='x', zorder=3)
    else:
        env.plot_target_failure_set(ax=ax, showCapture=False)
    
    if i == nRow-1:
        ax.set_xlabel('Exhaustive: {:.3f}'.format(exhaustiveValue), fontsize=fsz)

    #= Value
#     ax = axes[0]
#     v = env.get_value(agent.Q_network, 0., -0.2, -0.3, .75*np.pi, nx, ny)
#     im = ax.imshow(v.T, interpolation='none', extent=axStyle[0],
#         origin="lower", cmap='seismic', vmin=vmin, vmax=vmax, zorder=-1)
#     CS = ax.contour(xs, ys, v.T, levels=[0], colors='k', linewidths=2,
#         linestyles='dashed')
#     if i == nRow-1:
#         ax.set_xlabel('Value', fontsize=24)

    for ax in axes:
        env.plot_formatting(ax=ax)
        
    
fig.tight_layout()


# In[14]:


pick_idx = 0
test_idx = captureList[pick_idx]
captureInstant = captureInstantList[pick_idx]
print("For test [{}], the pursuer caught the evader at the {} step.".format(test_idx, captureInstant+1))

state = states[test_idx]
trajEvader = dictList[test_idx]['trajEvader']
trajPursuer = dictList[test_idx]['trajPursuer']
print(trajEvader.shape)

dist_list= np.array([np.linalg.norm(s1[:2]-s2[:2]) for s1, s2 in zip(trajEvader, trajPursuer)])
print(dist_list, env.capture_range)


captureFlag, captureInstant = checkCapture(env, trajEvader, trajPursuer)
print(captureFlag, captureInstant)


# ## DEFENDER VALIDATION --- POS

# In[15]:


validationFile = os.path.join(dataFolder, 'data', 'valDictPOS.npy')
valDictPos, successList, failureList, captureList, captureInstantList,    crossConstraintList, crossConstraintInstantList, unfinishedList = analyzeValidationResult(validationFile, env, verbose=True)

dictList = valDictPos['dictList']
exhaustiveValueList = valDictPos['exhaustiveValueList']
states = valDictPos['states']
stateIdxList = valDictPos['stateIdxList']

samplesDict = np.load(os.path.join(dataFolder, 'data', 'POS', 'samplesPOS.npy'), allow_pickle=True).tolist()


# In[16]:


agentList = [agent2]
axStyle = env.get_axes()
nx = 101
ny = 101
vmin = -1
vmax = 1
xs = np.linspace(env.bounds[0,0], env.bounds[0,1], nx)
ys = np.linspace(env.bounds[1,0], env.bounds[1,1], ny)

fsz=20
nRow = len(agentList)
nCol = 2
subfigSz = 4
figsize =  (nCol*subfigSz, nRow*subfigSz)

fig, axArray = plt.subplots(nRow, nCol, figsize=figsize, sharex=True, sharey=True)
pick_capture = False
pick_cross = False
pick_success = True

for i, agent in enumerate(agentList):
#     axes = axArray[i]
    axes = axArray
    
    pick_idx = 0
    if pick_capture:
        test_idx = captureList[pick_idx]
        endInstant = captureInstantList[pick_idx]
        print("For test [{}], the pursuer caught the evader at the {} step.".format(test_idx, endInstant))
    elif pick_cross:
        test_idx = crossConstraintList[pick_idx]
        endInstant = crossConstraintInstantList[pick_idx]
        print("For test [{}], the evader collided with the obstacle at the {} step.".format(test_idx, endInstant))
    elif pick_success:
        test_idx = successList[pick_idx]
        print("For test [{}], the evader succeeded.".format(test_idx))

    state = states[test_idx]
    trajEvader = dictList[test_idx]['trajEvader']
    trajPursuer = dictList[test_idx]['trajPursuer']
    exhaustiveValue = dictList[test_idx]['maxminV']

    #= Rollout
    ax = axes[0]
    _, minVs = env.plot_trajectories(agent.Q_network, states=[state],
            toEnd=False, ax=ax, lw=2, T=200, c=['#0abab5', 'k'])
    env.plot_target_failure_set(ax=ax, showCapture=False)
    
    if i == nRow-1:
        ax.set_xlabel('Policy Rollout: {:.3f}'.format(minVs[0]), fontsize=fsz)
        
    #= Exhaustive
    ax = axes[1]
    ax.plot(trajEvader[:endInstant, 0],  trajEvader[:endInstant, 1],  lw=2, c='#0abab5')
    ax.plot(trajPursuer[:endInstant, 0], trajPursuer[:endInstant, 1], lw=2, c='k')
    ax.scatter(trajEvader[0, 0],  trajEvader[0, 1],  s=48, c='#0abab5', zorder=3)
    ax.scatter(trajPursuer[0, 0], trajPursuer[0, 1], s=48, c='k', zorder=3)
    
    if pick_capture:
        xPursuer, yPursuer = trajPursuer[endInstant, :2]
        xEvader,  yEvader  = trajEvader[endInstant, :2]
        
        env.plot_target_failure_set(ax=ax, xPursuer=xPursuer, yPursuer=yPursuer)
        ax.scatter(xPursuer, yPursuer, s=48, c='b', marker='x', zorder=3)
        ax.scatter(xEvader,  yEvader,  s=48, c='b', marker='x', zorder=3)
    else:
        env.plot_target_failure_set(ax=ax, showCapture=False)
    
    if i == nRow-1:
        ax.set_xlabel('Exhaustive: {:.3f}'.format(exhaustiveValue), fontsize=fsz)

    #= Value
#     ax = axes[0]
#     v = env.get_value(agent.Q_network, 0., -0.2, -0.3, .75*np.pi, nx, ny)
#     im = ax.imshow(v.T, interpolation='none', extent=axStyle[0],
#         origin="lower", cmap='seismic', vmin=vmin, vmax=vmax, zorder=-1)
#     CS = ax.contour(xs, ys, v.T, levels=[0], colors='k', linewidths=2,
#         linestyles='dashed')
#     if i == nRow-1:
#         ax.set_xlabel('Value', fontsize=24)

    for ax in axes:
        env.plot_formatting(ax=ax)
        
    
fig.tight_layout()


# In[17]:


print(samplesDict.keys())
rolloutPolicyValueList = samplesDict['rolloutPolicyValueList']
print(rolloutPolicyValueList[102])


# In[18]:


env.simulate_one_trajectory(agent.Q_network, T=100, state=None)

