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
kp = 1.0 #  **** A MODIFIER EN TP ****

u = np.zeros((2, 1))

# main loop of simulation
for t in simulation.t:

	# computation for each robot of the fleet
    for i in range(0, fleet.nbOfRobots):
        
        
        for j in range(0, fleet.nbOfRobots):
        
            if communicationGraph.adjacencyMatrix[i, j] != 0 and j != i:
            
                u += (fleet.robot[j].state - fleet.robot[i].state)

    
		# control input of robot i
        fleet.robot[i].ctrl = kp * u  #  **** A COMPLETER EN TP **** #
		
        
    # store simulation data
    simulation.addDataFromFleet(fleet)
    # integrat motion over sampling period
    fleet.integrateMotion(Te)

# plot
plt.close('all')
simulation.plot()
#simulation.plotFleet()

