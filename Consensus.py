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


stop_simulation = False

def on_key(event):
    global stop_simulation
    if event.key == 'p':
        stop_simulation = True





# fleet definition
nbOfRobots = 6  
fleet = Robot.Fleet(nbOfRobots, dynamics='singleIntegrator2D')#, initState=initState)    


# random initial positions
np.random.seed(100)
for i in range(0, nbOfRobots):
    fleet.robot[i].state = 10*np.random.rand(2, 1)-5  # random init btw -5, +5

# communication graph
communicationGraph = Graph.Graph(nbOfRobots)
# adjacency matrix
communicationGraph.adjacencyMatrix = np.ones((6,6))

# np.array([[1, 0, 0, 0, 0, 0],
#                                                [0, 1, 0, 0, 0, 0],
#                                                [0, 0, 1, 0, 0, 0],
#                                                [0, 0, 0, 1, 0, 0],
#                                                [0, 0, 0, 0, 1, 0],
#                                                [0, 0, 0, 0, 0, 1]])

# plot communication graph
communicationGraph.plot(figNo=1)


# simulation parameters
Te = 0.01
simulation = Simulation.FleetSimulation(fleet, t0=0.0, tf=20.0, dt=Te)

# control gain for consensus
kp = 0.3 #  **** A MODIFIER EN TP ****
kr = 0.2
reference = np.array([[8.0], [8.0]])
safe_distance = 2
kr_avoid = 1.0

#animation
plt.ion()  # mode interactif
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)

# main loop of simulation
for t in simulation.t:
    
    if stop_simulation:
        print("Simulation arrêtée par l'utilisateur.")
        break


#        #proportional control law to common reference state
#        referenceState= np.array([[2.] , [1.] ])
#        for r in range(0, fleet.nbOfRobots):
#            fleet.robot[r].ctrl = kp* (referenceState - fleet.robot[r].state)

#
    print(type(fleet.robot[0].state))    
    fleet.robot[0].ctrl += kr * (reference - fleet.robot[0].state)
    
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
    

        
        fleet.robot[r].ctrl += kr * (fleet.robot[0] - fleet.robot[r].state)
        
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
    
    fig.canvas.draw()          # <<< indispensable dans Spyder
    fig.canvas.flush_events()  # <<< indispensable dans Spyder
    fig.canvas.mpl_connect('key_press_event', on_key)



		
        
   

# plot
# plt.close('all')
# simulation.plot(figNo=4)
# simulation.plotFleet(figNo=4, mod=200)

