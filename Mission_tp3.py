# -*- coding: utf-8 -*-
"""
Sample script for project

(c) S. Bertrand
"""


"""
Mission objectives: 
**********************
 - reach successively the three waypoints (black stars)
 - maintain the three robots in a triangle formation as much as possible:
 
                          #0
                       /   ^  \
                      /    |d  \         (with d=6m)
                    #1 <-> v <-> #2
                       d/2   d/2
                      
 - no motion through obstacles (gray rectangles)
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import Robot
import Simulation


# fleet definition
nbOfRobots = 3 
fleet = Robot.Fleet(nbOfRobots, dynamics='singleIntegrator2D')#, initState=initState)    


# initial positions
np.random.seed(100)
for i in range(0, nbOfRobots):
    fleet.robot[i].state = 10*np.random.rand(2, 1)-5  # random init btw -10, +10


# init simulation object
Te = 0.01
simulation = Simulation.FleetSimulation(fleet, t0=0.0, tf=70.0, dt=Te)


# WayPoints
WPListInit = [ np.array([[30],[0]]) , np.array([[30],[70]]),  np.array([[0],[70]])]


# obstacles
obstacle1 = patches.Rectangle((-10,20), 38, 20, color='grey', fill=True)
obstacle2 = patches.Rectangle((32,20), 8, 20, color='grey', fill=True)




# run simulation loop
for t in simulation.t:

    # *** TO BE COMPLETED ***
    
    # do not modify these two lines
    simulation.addDataFromFleet(fleet)
    fleet.integrateMotion(Te)



# plot
simulation.plot(figNo=2)
simulation.plotFleet(figNo=2, mod=150, links=True)
fig = plt.figure(2)
fig.axes[0].set_xlim(-10, 40)
fig.axes[0].set_ylim(-10, 80)
fig.axes[0].add_patch(obstacle1)
fig.axes[0].add_patch(obstacle2)
for wp in WPListInit:
    fig.axes[0].plot(wp[0], wp[1], color='k', marker='*', markersize=15)