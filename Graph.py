# -*- coding: utf-8 -*-
"""
Graph Class

(c) S. Bertrand
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
class Graph:
# =============================================================================
    
    # -------------------------------------------------------------------------
    def __init__(self, nbOfNodes, adjacencyMatrix = []):
    # -------------------------------------------------------------------------
        self.nbOfNodes = nbOfNodes
        if (len(adjacencyMatrix)==0):
            self.adjacencyMatrix = np.zeros((nbOfNodes,nbOfNodes))
            self.computeLaplaceMatrix()
        else:
            self.adjacencyMatrix = adjacencyMatrix
            self.computeLaplaceMatrix()


    # -------------------------------------------------------------------------
    def computeLaplaceMatrix(self):
    # -------------------------------------------------------------------------
        L = np.zeros((self.nbOfNodes, self.nbOfNodes))
        for i in range(0, self.nbOfNodes):
            l_ii = 0
            for j in range(0, self.nbOfNodes):
                if (i!=j):                
                    L[i,i] = - self.adjacencyMatrix[i,j]
                    l_ii += self.adjacencyMatrix[i,j]
            L[i,i] =l_ii
        self.LaplaceMatrix = L
        return L



    # -------------------------------------------------------------------------
    def getNeighbors(self, nodeNo):
    # -------------------------------------------------------------------------
        return np.where(self.adjacencyMatrix[nodeNo]!=0)[0]
        

    # -------------------------------------------------------------------------
    def plot(self, figNo=None):
    # -------------------------------------------------------------------------
        listOfNodes = np.arange(0, self.nbOfNodes)
        coordinates = np.zeros((2,self.nbOfNodes))
        if (figNo==None):        
            fig = plt.figure()
        else:
            fig = plt.figure(figNo)
        graph = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-15, 20), ylim=(-15, 15))                
            
        
        for node in listOfNodes:
            coordinates[0,node] = 10 * np.cos((2*np.pi/self.nbOfNodes)*node)
            coordinates[1,node] = 10* np.sin((2*np.pi/self.nbOfNodes)*node)
            graph.plot(coordinates[0,node], coordinates[1,node], marker = '8', linestyle="None", markersize=10, label=str(node))        

        for node in listOfNodes:         
            for neig in self.getNeighbors(node):
                graph.plot([coordinates[0,node], coordinates[0,neig]] , [coordinates[1,node], coordinates[1,neig]], color='k' )


        graph.legend()
        

# ====================== end of class Graph ===============================



# ============================== MAIN =========================================        
if __name__=='__main__':
# =============================================================================
    
    aMatrix = np.array([ [1, 0, 1] , [0, 1, 1] , [1, 1, 1]])
    g = Graph(3, adjacencyMatrix = aMatrix)    
    L = g.computeLaplaceMatrix()        
    neig = g.getNeighbors(1)
    g.plot()
    
    g2 = Graph(10, adjacencyMatrix = np.ones((10,10)))
    g2.plot()
    