import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle
from gym_reachability import gym_reachability  # Custom Gym env.
import gym

from KC_DQN.config import dqnConfig
from KC_DQN.DDQNPursuitEvasion import DDQNPursuitEvasion

tiffany = '#0abab5'


def plotTrajStep(state, env, agent, c=[tiffany, 'y'], lw=2, nx=101, ny=101, toEnd=False, T=100):
    """
    plotTrajStep [summary]

    Args:
        state ([type]): [description]
        env ([type]): [description]
        agent ([type]): [description]
        c (list, optional): [description]. Defaults to [tiffany, 'y'].
        lw (int, optional): [description]. Defaults to 2.
        nx (int, optional): [description]. Defaults to 101.
        ny (int, optional): [description]. Defaults to 101.
        toEnd (bool, optional): [description]. Defaults to False.
        T (int, optional): [description]. Defaults to 100.

    Returns:
        [type]: [description]
    
    Example:
        state = np.array([-.9, 0., 0., -.5, -.3, .75*np.pi])
        valueList, lxList, gxList = plotTrajStep(state, env, agent)
    """    
    trajEvader, trajPursuer, result, minV, info = env.simulate_one_trajectory(agent.Q_network, T=T, state=state, toEnd=toEnd)
    valueList = info['valueList']
    gxList = info['gxList']
    lxList = info['lxList']
    print('trajectory length is {:d}'.format(trajEvader.shape[0]))
    
    #== PLOT ==
    trajEvaderX = trajEvader[:,0]
    trajEvaderY = trajEvader[:,1]
    trajPursuerX = trajPursuer[:,0]
    trajPursuerY = trajPursuer[:,1]
    numCol = 5
    numRow = int(np.ceil(trajEvader.shape[0]/numCol))
    numAx = int(numRow*numCol)
    fig, axes = plt.subplots(numRow, numCol, figsize=(4*numCol, 4*numRow))

    for i in range(trajEvader.shape[0]):
        print("{:d}/{:d}".format(i+1, numAx), end='\r')
        rowIdx = int(i/numCol)
        colIdx = i % numCol
        if numRow > 1:
            ax = axes[rowIdx][colIdx]
        else:
            ax = axes[colIdx]
        xPursuer = trajPursuer[i,0]
        yPursuer = trajPursuer[i,1]
        thetaPursuer =trajPursuer[i,2]
        theta = trajEvader[i,2]

        cbarPlot = (i % numCol) == (numCol-1)
        ax.scatter(trajEvaderX[i], trajEvaderY[i], s=48, c=c[0], zorder=3)
        ax.plot(trajEvaderX[i:], trajEvaderY[i:], color=c[0],  linewidth=lw, zorder=2)
        ax.scatter(trajPursuerX[i], trajPursuerY[i], s=48, c=c[1], zorder=3)
        ax.plot(trajPursuerX[i:], trajPursuerY[i:], color=c[1],  linewidth=lw, zorder=2)

        env.plot_formatting(ax=ax)
        env.plot_target_failure_set(ax=ax, xPursuer=xPursuer, yPursuer=yPursuer)
        env.plot_v_values(  agent.Q_network, ax=ax, fig=fig, cbarPlot=cbarPlot,
                            theta=theta, xPursuer=xPursuer, yPursuer=yPursuer, thetaPursuer=thetaPursuer,
                            cmap='seismic', vmin=-.5, vmax=.5, nx=nx, ny=ny)
        ax.set_title(r'$v={:.3f}$'.format(valueList[i]), fontsize=16)
    return valueList, lxList, gxList


def pursuerResponse(env, agent, statePursuer, trajEvader):
    trajPursuer = []
    result = 0 # not finished

    valueList=[]
    gxList=[]
    lxList=[]
    for t in range(trajEvader.shape[0]):
        stateEvader = trajEvader[t]
        trajPursuer.append(statePursuer)
        state = np.concatenate((stateEvader, statePursuer), axis=0)
        donePursuer = not env.pursuer.check_within_bounds(statePursuer)

        g_x = env.safety_margin(state)
        l_x = env.target_margin(state)

        #= Rollout Record
        if t == 0:
            maxG = g_x
            current = max(l_x, maxG)
            minV = current
        else:
            maxG = max(maxG, g_x)
            current = max(l_x, maxG)
            minV = min(current, minV)

        valueList.append(minV)
        gxList.append(g_x)
        lxList.append(l_x)

        #= Dynamics
        stateTensor = torch.FloatTensor(state).to(env.device)
        with torch.no_grad():
            state_action_values = agent.Q_network(stateTensor)
        Q_mtx = state_action_values.reshape(env.numActionList[0], env.numActionList[1])
        pursuerValues, colIndices = Q_mtx.max(dim=1)
        minmaxValue, rowIdx = pursuerValues.min(dim=0)
        colIdx = colIndices[rowIdx]

        # If cars are within the boundary, we update their states according to the controls
        if not donePursuer:
            uPursuer = env.pursuer.discrete_controls[colIdx]
            statePursuer = env.pursuer.integrate_forward(statePursuer, uPursuer)

    trajPursuer = np.array(trajPursuer)
    info = {'valueList':valueList, 'gxList':gxList, 'lxList':lxList}
    return trajPursuer, result, minV, info


def evaderResponse(env, agent, state, actionSeq, maxLength=40):
    numPursuerStep = actionSeq.shape[0]
    resPeriod = int(np.ceil(maxLength/numPursuerStep))
    stateEvader  = state[:3]
    statePursuer = state[3:]
    trajPursuer = [statePursuer]
    trajEvader = [stateEvader]
    valueList = []
    gxList = []
    lxList = []
    pursuerActionIdx = 0

    for t in range(maxLength):
        
        state = np.concatenate((stateEvader, statePursuer), axis=0)
        doneEvader = not env.evader.check_within_bounds(stateEvader)
        donePursuer = not env.pursuer.check_within_bounds(statePursuer)

        g_x = env.safety_margin(state)
        l_x = env.target_margin(state)

        #= Rollout Record
        if t == 0:
            maxG = g_x
            current = max(l_x, maxG)
            minV = current
        else:
            maxG = max(maxG, g_x)
            current = max(l_x, maxG)
            minV = min(current, minV)

        valueList.append(minV)
        gxList.append(g_x)
        lxList.append(l_x)

        #= Dynamics
        stateTensor = torch.FloatTensor(state).to(env.device)
        with torch.no_grad():
            state_action_values = agent.Q_network(stateTensor)
        Q_mtx = state_action_values.reshape(env.numActionList[0], env.numActionList[1])
        pursuerValues, colIndices = Q_mtx.max(dim=1)
        minmaxValue, rowIdx = pursuerValues.min(dim=0)
        colIdx = colIndices[rowIdx]

        # If cars are within the boundary, we update their states according to the controls
        if not doneEvader:
            uEvader = env.evader.discrete_controls[rowIdx]
            stateEvader = env.evader.integrate_forward(stateEvader, uEvader)
        if not donePursuer:
            actionIdx = actionSeq[pursuerActionIdx]
            uPursuer = env.pursuer.discrete_controls[actionIdx]
            statePursuer = env.pursuer.integrate_forward(statePursuer, uPursuer)

        trajPursuer.append(statePursuer)
        trajEvader.append(stateEvader)
        if (t+1) % resPeriod == 0:
            pursuerActionIdx += 1

    trajEvader = np.array(trajEvader)
    trajPursuer = np.array(trajPursuer)
    info = {'valueList':valueList, 'gxList':gxList, 'lxList':lxList}
    return trajEvader, trajPursuer, minV, info


def validateEvaderPolicy(env, agent, state, maxLength=40, numPursuerStep=10):
    actionSet= np.empty(shape=(env.numActionList[1], numPursuerStep), dtype=int)
    for i in range(numPursuerStep):
        actionSet[:, i] = np.arange(env.numActionList[1])

    shapeTmp = np.ones(numPursuerStep, dtype=int)*env.numActionList[1]
    rolloutResult = np.empty(shape=shapeTmp, dtype=int)
    it = np.nditer(rolloutResult, flags=['multi_index'])
    responseDict={'state':state, 'maxLength':maxLength, 
        'numPursuerStep':numPursuerStep}
    flag = True
    while not it.finished:
        idx = it.multi_index
        actionSeq = actionSet[idx, np.arange(numPursuerStep)]
        print(actionSeq, end='\r')
        trajEvader, trajPursuer, minV, _ = evaderResponse(
            env, agent, state, actionSeq, maxLength)
        info = {'trajEvader':trajEvader, 'trajPursuer':trajPursuer, 'minV':minV}
        responseDict[idx] = info
        it.iternext()
        if flag:
            maxminV = minV
            maxminIdx = idx
            flag = False
        elif minV > maxminV:
            maxminV = minV
            maxminIdx = idx
    responseDict['maxminV'] = maxminV
    responseDict['maxminIdx'] = maxminIdx
    return responseDict


def loadEnv(args, verbose=True):
    env_name = "dubins_car_pe-v0"
    if args.forceCPU:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(env_name, device=device, mode='RA', doneType='toEnd')

    if verbose:
        print("\n== Environment Information ==")
        env.report()
        print()
    return env


def loadAgent(args, configFile, device, stateNum, actionNum, numActionList,
    verbose=True):
    if verbose:
        print("\n== Agent Information ==")
    with open(configFile, 'rb') as handle:
        tmpConfig = pickle.load(handle)
    CONFIG = dqnConfig()
    for key, value in tmpConfig.__dict__.items():
        CONFIG.__dict__[key] = tmpConfig.__dict__[key]
    CONFIG.DEVICE = device
    CONFIG.SEED = 0

    dimList = [stateNum] + CONFIG.ARCHITECTURE + [actionNum]
    agent = DDQNPursuitEvasion(CONFIG, numActionList, dimList, CONFIG.ACTIVATION)
    modelFile = '{:s}/model-{:d}.pth'.format(args.modelFolder+'/model', 4000000)
    agent.restore(modelFile)

    if verbose:
        print(vars(CONFIG))
        print('agent\'s device:', agent.device)

    return agent