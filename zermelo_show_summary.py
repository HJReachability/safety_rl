# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

# This is the summary of different agent types, cost signal types, done signal
# types and gamma choices.
# The subtitles are composed of <agent_type>_<done_type>[_<cost type>]_<gamma>.
# agent types:
    # RA (reach-avoid)
    # L (Lagrange)
# done types:
    # E (to end, stop the episode when going outside the world)
    # F (fail, stop the episode when entering failure set)
    # TF (target-fail, stop the episode when entering failure or target set)
    # E: initial states are sampled over the whole environment
    # F and TF: initial states are sampled outside the failure set
# cost types:
    # D (dense, ell + g)
    # S (sparse, +1: failure; -1: target)
    # no specification (reach-avoid)
# gamma choices: 9 (0.9), 95 (0.95), 999 (0.999), 9999(0.9999)

from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning)

from gym_reachability import gym_reachability  # Custom Gym env.
import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle
import argparse
import os

from KC_DQN.DDQNSingle import DDQNSingle
from KC_DQN.utils import save_obj, load_obj

#== ARGS ==
parser = argparse.ArgumentParser()
parser.add_argument("-n",  "--n", default=101,  type=int)
args = parser.parse_args()
print(args)


def getModelInfo(path):
    dataFolder = os.path.join('scratch', path)
    modelFolder = os.path.join(dataFolder, 'model')

    #= pickle
    picklePath = os.path.join(modelFolder, 'CONFIG.pkl')
    with open(picklePath, 'rb') as fp:
        config = pickle.load(fp)
    config.DEVICE = 'cpu'

    #= model idx
    trainPath = os.path.join(dataFolder, 'train')
    f = load_obj(trainPath)
    trainProgress = f['trainProgress']
    idx = np.argmax(trainProgress[:, 0]) + 1
    successRate = np.amax(trainProgress[:, 0]) 
    print('We pick model with success rate-{:.3f}'.format(successRate))

    dimList = [stateDim] + config.ARCHITECTURE + [actionNum]
    return dataFolder, config, dimList, idx


#== ENV ==
print("\n== ENVIRONMENT ==")
env_name = "zermelo_show-v0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make(env_name, device=device, mode='RA', doneType='toEnd', thickness=0.1, sample_inside_obs=True)

stateDim = env.state.shape[0]
actionNum = env.action_space.n
action_list = np.arange(actionNum)
print("State Dimension: {:d}, ActionSpace Dimension: {:d}".format(stateDim, actionNum))


#== AGENT ==
print("\n== AGENT ==")
dataFolder, CONFIG, dimList, idx = getModelInfo(os.path.join('show', 'RA', '9999-s-toEnd'))
agent_RA_E = DDQNSingle(CONFIG, actionNum, action_list, dimList=dimList, mode='RA', verbose=False)
agent_RA_E.restore(idx*25000, dataFolder)

dataFolder, CONFIG, dimList, idx = getModelInfo(os.path.join('show', 'RA', '9999-s-fail'))
agent_RA_F = DDQNSingle(CONFIG, actionNum, action_list, dimList=dimList, mode='RA', verbose=False)
agent_RA_F.restore(idx*25000, dataFolder)

dataFolder, CONFIG, dimList, idx = getModelInfo(os.path.join('show', 'RA', '9999-s-TF'))
agent_RA_TF = DDQNSingle(CONFIG, actionNum, action_list, dimList=dimList, mode='RA', verbose=False)
agent_RA_TF.restore(idx*25000, dataFolder)

dataFolder, CONFIG, dimList, idx = getModelInfo(os.path.join('show', 'RA', '999-s-toEnd'))
agent_RA_E_999 = DDQNSingle(CONFIG, actionNum, action_list, dimList=dimList, mode='RA', verbose=False)
agent_RA_E_999.restore(idx*25000, dataFolder)

dataFolder, CONFIG, dimList, idx = getModelInfo(os.path.join('show', 'RA', '999-s-fail'))
agent_RA_F_999 = DDQNSingle(CONFIG, actionNum, action_list, dimList=dimList, mode='RA', verbose=False)
agent_RA_F_999.restore(idx*25000, dataFolder)

dataFolder, CONFIG, dimList, idx = getModelInfo(os.path.join('show', 'RA', '99-s-toEnd'))
agent_RA_E_99 = DDQNSingle(CONFIG, actionNum, action_list, dimList=dimList, mode='RA', verbose=False)
agent_RA_E_99.restore(idx*25000, dataFolder)

dataFolder, CONFIG, dimList, idx = getModelInfo(os.path.join('show', 'RA', '99-s-fail'))
agent_RA_F_99 = DDQNSingle(CONFIG, actionNum, action_list, dimList=dimList, mode='RA', verbose=False)
agent_RA_F_99.restore(idx*25000, dataFolder)

dataFolder, CONFIG, dimList, idx = getModelInfo(os.path.join('show', 'lagrange', '95-s-TF-sparse'))
agent_L_TF_S = DDQNSingle(CONFIG, actionNum, action_list, dimList=dimList, mode='normal', verbose=False)
agent_L_TF_S.restore(idx*25000, dataFolder)

dataFolder, CONFIG, dimList, idx = getModelInfo(os.path.join('show', 'lagrange', '95-s-TF-dense'))
agent_L_TF_D = DDQNSingle(CONFIG, actionNum, action_list, dimList=dimList, mode='normal', verbose=False)
agent_L_TF_D.restore(idx*25000, dataFolder)

dataFolder, CONFIG, dimList, idx = getModelInfo(os.path.join('show', 'lagrange', '9-s-TF-sparse'))
agent_L_TF_S_9 = DDQNSingle(CONFIG, actionNum, action_list, dimList=dimList, mode='normal', verbose=False)
agent_L_TF_S_9.restore(idx*25000, dataFolder)

dataFolder, CONFIG, dimList, idx = getModelInfo(os.path.join('show', 'lagrange', '99-s-TF-sparse'))
agent_L_TF_S_99 = DDQNSingle(CONFIG, actionNum, action_list, dimList=dimList, mode='normal', verbose=False)
agent_L_TF_S_99.restore(idx*25000, dataFolder)

dataFolder, CONFIG, dimList, idx = getModelInfo(os.path.join('show', 'lagrange', '95-s-toEnd-sparse'))
agent_L_E_S = DDQNSingle(CONFIG, actionNum, action_list, dimList=dimList, mode='normal', verbose=False)
agent_L_E_S.restore(idx*25000, dataFolder)

dataFolder, CONFIG, dimList, idx = getModelInfo(os.path.join('show', 'lagrange', '95-s-fail-sparse'))
agent_L_F_S = DDQNSingle(CONFIG, actionNum, action_list, dimList=dimList, mode='normal', verbose=False)
agent_L_F_S.restore(idx*25000, dataFolder)

agentList = [   agent_RA_E, agent_RA_F, agent_RA_TF,
                agent_RA_E_999, agent_RA_F_999, agent_RA_E_99, agent_RA_F_99,
                agent_L_TF_S, agent_L_TF_D,
                agent_L_TF_S_9, agent_L_TF_S_99,
                agent_L_E_S, agent_L_F_S]
nList = [   'RA_E_9999', 'RA_F_9999', 'RA_TF_9999',
            'RA_E_999', 'RA_F_999', 'RA_E_99', 'RA_F_99', 
            'L_TF_S_95', 'L_TF_D_95',
            'L_TF_S_9', 'L_TF_S_99',
            'L_E_S_95', 'L_F_S_95']

numAgent = len(agentList)


#== ROLLOUT RA ==
print("\n== ROLLOUT REACH-AVOID ==")
nx=args.n
ny=args.n
print("Test on {}x{} grids".format(nx, ny))
xs = np.linspace(env.bounds[0,0], env.bounds[0,1], nx)
ys = np.linspace(env.bounds[1,0], env.bounds[1,1], ny)

resultMtxList  = np.empty((len(agentList), nx, ny), dtype=int)
actDistMtxList = np.empty((len(agentList), nx, ny), dtype=int)

for i, agent in enumerate(agentList):
    print('== {}/{} =='.format(i, numAgent))
    resultMtx  = np.empty((nx, ny), dtype=int)
    actDistMtx = np.empty((nx, ny), dtype=int)

    it = np.nditer(resultMtx, flags=['multi_index'])

    while not it.finished:
        idx = it.multi_index
        print(idx, end='\r')
        x = xs[idx[0]]
        y = ys[idx[1]]

        state = np.array([x, y])
        stateTensor = torch.FloatTensor(state).unsqueeze(0)
        action_index = agent.Q_network(stateTensor).min(dim=1)[1].item()
        actDistMtx[idx] = action_index

        _, _, result = env.simulate_one_trajectory(agent.Q_network, T=250, state=state, toEnd=False)
        resultMtx[idx] = result
        it.iternext()
    
    resultMtxList[i] = resultMtx
    actDistMtxList[i] = actDistMtx
    print()

outFolder = os.path.join('scratch', 'show', 'result')
os.makedirs(outFolder, exist_ok=True)

recordDict = {}
recordDict['resultMtxList'] = resultMtxList
recordDict['actDistMtxList'] = actDistMtxList
filePath = os.path.join(outFolder, 'record')
save_obj(recordDict, filePath)


#== FIGURE ==
print("\n== FIGURE ==")
nRow = 3
nCol = numAgent

fig, axArray = plt.subplots(nRow, nCol, figsize=(4*nCol, 4*nRow), sharex=True, sharey=True)
axStyle = env.get_axes()

for i, agent in enumerate(agentList):
    print(i, end='\r')
    resultMtx = resultMtxList[i]
    actDistMtx = actDistMtxList[i]
    axArray[0][i].set_title(nList[i], fontsize=28)
    
    if i <= 3:
        vmax = 4
        vmin = -vmax
    elif i == 5:
        vmax = 400
        vmin = -100
    else:
        vmax = 10
        vmin = -vmax

    #= Action
    ax = axArray[2][i]
    im = ax.imshow(actDistMtx.T, interpolation='none', extent=axStyle[0],
        origin="lower", cmap='seismic', vmin=0, vmax=2, zorder=-1)
    
    #= Rollout
    ax = axArray[1][i]
    im = ax.imshow(resultMtx.T != 1, interpolation='none', extent=axStyle[0],
        origin="lower", cmap='seismic', vmin=0, vmax=1, zorder=-1)
    env.plot_trajectories(agent.Q_network, states=env.visual_initial_states, toEnd=True, ax=ax, c='w', lw=1.5)

    #= Value
    ax = axArray[0][i]
    _, _, v = env.get_value(agent.Q_network, nx, ny)
    im = ax.imshow(v.T, interpolation='none', extent=axStyle[0],
        origin="lower", cmap='seismic', vmin=vmin, vmax=vmax, zorder=-1)
    CS = ax.contour(xs, ys, v.T, levels=[0], colors='k', linewidths=2,
        linestyles='dashed')

for axes in axArray:
    for ax in axes:
        env.plot_target_failure_set(ax=ax)
        env.plot_reach_avoid_set(ax=ax)
        env.plot_formatting(ax=ax)

axArray[0][0].set_ylabel('Value', fontsize=24)
axArray[1][0].set_ylabel('Rollout', fontsize=24)
axArray[2][0].set_ylabel('Action', fontsize=24) 
fig.tight_layout()
figurePath = os.path.join(outFolder, 'value_rollout_action.png')
fig.savefig(figurePath)
figurePath = os.path.join(outFolder, 'value_rollout_action.eps')
fig.savefig(figurePath)