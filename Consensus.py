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
communicationGraph.adjacencyMatrix = np.ones((nbOfRobots,nbOfRobots))

# plot communication graph
communicationGraph.plot(figNo=1)


# simulation parameters
Te = 0.01
simulation = Simulation.FleetSimulation(fleet, t0=0.0, tf=20.0, dt=Te)

# control gain for consensus
kp = 0.4 #  **** A MODIFIER EN TP ****

u = np.zeros((2, 1))

# main loop of simulation
for t in simulation.t:

#        #proportional control law to common reference state
#        referenceState= np.array([[2.] , [1.] ])
#        for r in range(0, fleet.nbOfRobots):
#            fleet.robot[r].ctrl = kp* (referenceState - fleet.robot[r].state)

    # consensus
    for r in range(0, fleet.nbOfRobots):
        fleet.robot[r].ctrl = np.zeros((2,1))
        for n in range(0, fleet.nbOfRobots):
            fleet.robot[r].ctrl += kp* (fleet.robot[n].state - fleet.robot[r].state) / fleet.nbOfRobots

		
        
    # store simulation data
    simulation.addDataFromFleet(fleet)
    # integrat motion over sampling period
    fleet.integrateMotion(Te)

# plot
plt.close('all')
simulation.plot(figNo=4)
simulation.plotFleet(figNo=4, mod=200)

