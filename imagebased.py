import matplotlib.pyplot as pl
import matplotlib
import numpy as np

import agent as ag

# Happy pdf for a happy submission without complains in paperplaza, arxiv, etc
font = {'size'   : 20}

matplotlib.rc('font', **font)

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True

# Simulation parameters
area = 100 # area X area initial positions
scale = 1 # plotting scale*area
tf = 100
dt = 5e-3
time = np.linspace(0, tf, tf/dt)
it = 0;
frames = 1/dt # The bigger, the faster the simulation
agentcolor = ['r', 'g', 'b', 'k', 'm', 'c']

# Incidice matrix from undirected graph
Btwo = np.array([[1], [-1]])
desired_dotp_two = np.array([0.8])

Btriang = np.array([[1, 0, -1],[-1, 1, 0],[0, -1, 1]]) #row agents, columns edges
desired_dotp_triang = np.array([0.85, 0.9, 0.95]) # edge 1, 2, 3...

B = Btriang
d = desired_dotp_triang

list_agents = []
num_agents = B.shape[0]
gain = 3

# Data log
P_h = np.zeros((time.size, 2*num_agents))
E_h = np.zeros((time.size, d.size))

for i in range(num_agents):
    list_nei = []
    for j in np.nonzero(B[i,:])[0]:
        for jj in np.nonzero(B[:,j])[0]:
            if jj != i:
                nei = [jj, d[j]]
                list_nei.append(nei)

    agent = ag.agent(i, agentcolor[i], list_nei, area*np.random.rand(2)-area/2, np.array([5, 0]), gain, time.size)
    list_agents.append(agent)

# Data log

pl.close("all")
pl.ion()
fig = pl.figure(0)

for t in time:
    for ag in list_agents:
        ag.step_Euler(ag.control_image_based(list_agents), dt)

    # Animation
    if it%frames == 0:
        pl.clf()

        for ag in list_agents:
            pl.plot(ag.p[0], ag.p[1], marker = 'o', color=ag.color)
            pl.plot([ag.p[0]+ag.l[0], ag.p[0]-ag.l[0]], [ag.p[1]+ag.l[1], ag.p[1]-ag.l[1]], marker = '|', markeredgecolor = ag.color, color=ag.color, markeredgewidth=10)
            for nei in ag.list_nei:
                pl.plot([ag.p[0], list_agents[nei[0]].p[0]-list_agents[nei[0]].l[0]], [ag.p[1], list_agents[nei[0]].p[1]-list_agents[nei[0]].l[1]], color=ag.color, ls='--')
                pl.plot([ag.p[0], list_agents[nei[0]].p[0]+list_agents[nei[0]].l[0]], [ag.p[1], list_agents[nei[0]].p[1]+list_agents[nei[0]].l[1]], color=ag.color, ls='--')

        pl.xlim(-scale*area, scale*area)
        pl.ylim(-scale*area, scale*area)
        pl.grid()
        pl.pause(0.001)
        pl.draw()

    it+=1


# post-process

for ag in list_agents:
    pl.plot(ag.log_P[:,0], ag.log_P[:,1], ag.color)

fig_count = 1
fig = pl.figure(fig_count)

for ag in list_agents:
    for ni in range(len(ag.list_nei)):
        pl.plot(time, ag.log_E[:,ni])
        pl.title("Error signals for agent %i"%ag.label)
    fig_count = fig_count + 1
    pl.grid()
    if ag.label != len(list_agents)-1:
        fig = pl.figure(fig_count)

pl.pause(0)
