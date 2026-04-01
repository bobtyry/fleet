# -*- coding: utf-8 -*-
"""
Multi-agent consensus simulation

(c) S. Bertrand
"""


import numpy as np
import Robot
import Graph
import Simulation
import matplotlib.pyplot as plt

#%%
stop_simulation = False
def on_key(event):
    global stop_simulation
    if event.key == 'p':
        stop_simulation = True


#%%


# fleet definition
nbOfRobots = 6  
fleet = Robot.Fleet(nbOfRobots, dynamics='singleIntegrator2D')#, initState=initState)    


# random initial positions
np.random.seed(100)
for i in range(0, nbOfRobots):
    fleet.robot[i].state = 20*np.random.rand(2, 1)-10  # random init btw -5, +5

# communication graph
communicationGraph = Graph.Graph(nbOfRobots)
# adjacency matrix
communicationGraph.adjacencyMatrix = np.eye(6)
for i in range(1,nbOfRobots):
    communicationGraph.adjacencyMatrix[i,0]=1
    communicationGraph.adjacencyMatrix[0,i]=1
    
print(communicationGraph.adjacencyMatrix)
#%%

# np.array([[1, 0, 0, 0, 0, 0],
#                                                [0, 1, 0, 0, 0, 0],
#                                                [0, 0, 1, 0, 0, 0],
#                                                [0, 0, 0, 1, 0, 0],
#                                                [0, 0, 0, 0, 1, 0],
#                                                [0, 0, 0, 0, 0, 1]])

# plot communication graph

max_degree = 4  # 1 liaison vers robot 0 + 2 voisins

for i in range(1, nbOfRobots):

    # calcul des distances
    liaison = {}
    for j in range(nbOfRobots):
        if j != i:
            dist = np.linalg.norm(fleet.robot[i].state - fleet.robot[j].state)
            liaison[j] = dist

    # tri par distance
    liaison_trie = sorted(liaison.items(), key=lambda x: x[1])

    # degré actuel de chaque robot
    degree = np.sum(communicationGraph.adjacencyMatrix, axis=1)

    # sélection des deux voisins admissibles
    deux_plus_proche = []
    for key, dist in liaison_trie:
        if degree[key] < max_degree:
            deux_plus_proche.append((key, dist))
        if len(deux_plus_proche) == 2:
            break

    # ajout des liaisons symétriques
    for key, dist in deux_plus_proche:
        communicationGraph.adjacencyMatrix[i, key] = 1
        communicationGraph.adjacencyMatrix[key, i] = 1

        
            
communicationGraph.plot(figNo=1)
print(communicationGraph.adjacencyMatrix)
#%%

# simulation parameters
Te = 0.01
simulation = Simulation.FleetSimulation(fleet, t0=0.0, tf=20.0, dt=Te)

# control gain for consensus
kp = 0.3 #  **** A MODIFIER EN TP ****
kr = 0.2
reference = np.array([[8.0], [8.0]])
safe_distance = 2
kr_avoid = 1.0
marge = 1

#animation
plt.ion()  # mode interactif
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)

#flag
etape_1 = True
etape_2 = False
Etape_f = False

#consigne
consigne = []
check = 0
for i in range(nbOfRobots-1):
    consigne.append(False)

# main loop of simulation

for t in simulation.t:
    
    if stop_simulation:
        print("Simulation arrêtée par l'utilisateur.")
        break

    if etape_1:
        
        for r in range(1, fleet.nbOfRobots):
            fleet.robot[r].ctrl = np.zeros((2,1))
            for n in range(0, fleet.nbOfRobots):
                if n != r and (communicationGraph.adjacencyMatrix[n,r]==1):
                    
                    d = np.linalg.norm(fleet.robot[n].state - fleet.robot[r].state)

                    # consensus si assez loin
                    if d > 3:
                        fleet.robot[r].ctrl += kp * (fleet.robot[n].state - fleet.robot[r].state) / fleet.nbOfRobots

                    # répulsion si trop proche
                    if d < safe_distance:
                        direction = fleet.robot[r].state - fleet.robot[n].state
                        fleet.robot[r].ctrl += kr_avoid * direction / (d + 1e-6)
        
            
            
            fleet.robot[r].ctrl += kr * (fleet.robot[0].state - fleet.robot[r].state)
            if (np.linalg.norm(fleet.robot[0].state - fleet.robot[r].state))<safe_distance + marge:
                consigne[r-1] = True
            check= 0
            for i in consigne:
                if i:
                    check = check + 1
            if check == 5:
                etape_1 = False
                etape_2 = True
            
    if etape_2:
        
        fleet.robot[0].ctrl = np.zeros((2,1))
        fleet.robot[0].ctrl += kr * (reference - fleet.robot[0].state)    
        
        for r in range(1, fleet.nbOfRobots):
            fleet.robot[r].ctrl = fleet.robot[0].ctrl
            

    # store simulation data
    simulation.addDataFromFleet(fleet)
    # integrat motion over sampling period
    fleet.integrateMotion(Te)
        
    # --- ANIMATION ---     
    ax.clear()
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    for i in range(nbOfRobots):
        x = fleet.robot[i].state[0]
        y = fleet.robot[i].state[1]
        ax.plot(x, y, 'o', markersize=10)
        ax.text(x + 0.2, y + 0.2, str(i), fontsize=12, color='black')

    
    fig.canvas.draw()          # <<< indispensable dans Spyder
    fig.canvas.flush_events()  # <<< indispensable dans Spyder
    fig.canvas.mpl_connect('key_press_event', on_key)



		
        
   

# plot
# plt.close('all')
# simulation.plot(figNo=4)
# simulation.plotFleet(figNo=4, mod=200)

