# -*- coding: utf-8 -*-
"""
Multi-agent consensus simulation
(c) S. Bertrand — version obstacles fixes
"""

import numpy as np
import Robot
import Graph
import Simulation
import matplotlib.pyplot as plt
import matplotlib.patches as patches

stop_simulation = False

def on_key(event):
    global stop_simulation
    if event.key == 'p':
        stop_simulation = True

# -------------------------------------------------------
# CALCUL DU POINT LE PLUS PROCHE POUR UN RECTANGLE
# -------------------------------------------------------
def closest_point_on_rect(px, py, rx, ry, w, h):
    cx = np.clip(px, rx, rx + w)
    cy = np.clip(py, ry, ry + h)
    return np.array([cx, cy])

# -------------------------------------------------------
# ÉVITEMENT OBSTACLES (champ tangentiel)
# -------------------------------------------------------
def compute_avoidance_ctrl(robot_state, rects, detection_radius, k_avoid=1.5):
    ctrl = np.zeros((2, 1))
    pos = robot_state.flatten()
    obstacle_points = []

    for (rx, ry, w, h) in rects:
        obs = closest_point_on_rect(pos[0], pos[1], rx, ry, w, h)
        d = float(np.linalg.norm(pos - obs))
        obstacle_points.append((obs, max(d, 0.01)))

    for (obs, d) in obstacle_points:
        if d < detection_radius:
            repulse = (pos - obs) / (d + 1e-6)
            tangent_cw  = np.array([ repulse[1], -repulse[0]])
            tangent_ccw = np.array([-repulse[1],  repulse[0]])
            intensity = k_avoid * (1.0 / d - 1.0 / detection_radius)
            intensity = max(intensity, 0.0)
            alpha = 0.4
            beta  = 0.6
            goal_dir = np.array([8.0, 8.0]) - pos
            tangent = tangent_ccw if np.dot(tangent_ccw, goal_dir) >= np.dot(tangent_cw, goal_dir) else tangent_cw
            avoid_vec = alpha * repulse + beta * tangent
            ctrl += intensity * avoid_vec.reshape(2, 1)

    return ctrl

# -------------------------------------------------------
# FLOTTE
# -------------------------------------------------------
nbOfRobots = 6
fleet = Robot.Fleet(nbOfRobots, dynamics='singleIntegrator2D')

np.random.seed(100)
for i in range(nbOfRobots):
    fleet.robot[i].state = 10 * np.random.rand(2, 1) - 5

# -------------------------------------------------------
# RÉFÉRENCE
# -------------------------------------------------------
reference_points = [
    np.array([[30.0], [0.0]]),
    np.array([[30.0], [70.0]]),
    np.array([[0.0], [70.0]])
]
ref_index = 0

DETECTION_RADIUS = 2.0

# -------------------------------------------------------
# OBSTACLES FIXES
# -------------------------------------------------------
rect_obstacles = [
    (-10, 20, 38, 20),
    (32, 20, 8, 20)
]

# -------------------------------------------------------
# GRAPHE DE COMMUNICATION
# -------------------------------------------------------
communicationGraph = Graph.Graph(nbOfRobots)
communicationGraph.adjacencyMatrix = np.ones((nbOfRobots, nbOfRobots))
communicationGraph.plot(figNo=1)

# -------------------------------------------------------
# SIMULATION
# -------------------------------------------------------
Te = 0.01
simulation = Simulation.FleetSimulation(fleet, t0=0.0, tf=20.0, dt=Te)

kp = 0.3
kr = 0.2
safe_distance = 1.0
kr_avoid = 1.0
k_obstacle_avoid = 1.5
vmax_effective = 10.0  # vitesse réelle pour ref1 et ref2

# -------------------------------------------------------
# FIGURE
# -------------------------------------------------------
plt.close('all')
plt.ion()
fig, ax = plt.subplots(figsize=(7,7))
ax.set_aspect('equal')
fig.canvas.mpl_connect('key_press_event', on_key)
plt.show(block=False)

# -------------------------------------------------------
# BOUCLE PRINCIPALE
# -------------------------------------------------------
for t in simulation.t:
    if stop_simulation:
        print("Simulation arrêtée par l'utilisateur.")
        break

    for r in range(nbOfRobots):
        fleet.robot[r].ctrl = np.zeros((2, 1))
        pos = fleet.robot[r].state.flatten()

        # Calcul distance à la référence
        target = reference_points[ref_index].flatten()
        dir_vec = target - pos
        dist = np.linalg.norm(dir_vec)

        # Changement de référence si proche
        if dist < 0.5 and ref_index < len(reference_points)-1:
            ref_index += 1
            target = reference_points[ref_index].flatten()
            dir_vec = target - pos
            dist = np.linalg.norm(dir_vec)

        # 1) Consensus + répulsion inter-robots
        for n in range(nbOfRobots):
            if n != r and communicationGraph.adjacencyMatrix[n, r] == 1:
                d = float(np.linalg.norm(fleet.robot[n].state - fleet.robot[r].state))
                if d > 3:
                    fleet.robot[r].ctrl += kp * (fleet.robot[n].state - fleet.robot[r].state) / nbOfRobots
                if d < safe_distance:
                    direction = fleet.robot[r].state - fleet.robot[n].state
                    fleet.robot[r].ctrl += kr_avoid * direction / (d + 1e-6)

        # 2) Attraction vers la référence
        if ref_index < 2:  # ref1 et ref2 → vitesse constante
            if dist > 0.01:
                dir_unit = dir_vec / dist
                fleet.robot[r].ctrl = (vmax_effective * dir_unit).reshape(2,1)
        else:  # ref3 → vitesse proportionnelle
            fleet.robot[r].ctrl += kr * dir_vec.reshape(2,1)

        # 3) Évitement obstacles
        fleet.robot[r].ctrl += compute_avoidance_ctrl(
            fleet.robot[r].state,
            rect_obstacles,
            DETECTION_RADIUS,
            k_avoid=k_obstacle_avoid
        )

    # Intégration du mouvement
    simulation.addDataFromFleet(fleet)
    fleet.integrateMotion(Te)

    # ---------------------------------------------------
    # AFFICHAGE
    # ---------------------------------------------------
    ax.clear()
    ax.set_xlim(-15, 45)
    ax.set_ylim(-10, 80)
    ax.set_title(f't = {t:.2f} s  |  [p] pour arrêter')
    ax.grid(True, linestyle='--', alpha=0.4)

    # obstacles gris
    for (rx, ry, w, h) in rect_obstacles:
        ax.add_patch(patches.Rectangle((rx, ry), w, h, color='grey', alpha=0.8))

    # référence
    ax.plot(target[0], target[1], '*', color='red', markersize=15, label='Référence')

    # robots
    for i in range(nbOfRobots):
        x = float(fleet.robot[i].state[0,0])
        y = float(fleet.robot[i].state[1,0])
        ax.plot(x, y, 'o', markersize=10, label=f'R{i}')

    ax.legend(loc='upper left', fontsize=7)
    fig.canvas.draw_idle()
    plt.pause(0.001)