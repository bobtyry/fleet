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
import matplotlib.patches as patches

stop_simulation = False

def on_key(event):
    global stop_simulation
    if event.key == 'p':
        stop_simulation = True

# -------------------------------------------------------
# GÉNÉRATION ALÉATOIRE DES OBSTACLES
# -------------------------------------------------------
def generate_obstacles(fleet, reference, nb_obstacles=4, seed=None):
    rng = np.random.default_rng(seed)

    all_x = [fleet.robot[i].state[0, 0] for i in range(fleet.nbOfRobots)]
    all_y = [fleet.robot[i].state[1, 0] for i in range(fleet.nbOfRobots)]
    all_x.append(float(reference[0, 0]))
    all_y.append(float(reference[1, 0]))

    margin = 1.0
    x_min, x_max = min(all_x) - margin, max(all_x) + margin
    y_min, y_max = min(all_y) - margin, max(all_y) + margin

    circles, rects, points = [], [], []
    types = rng.choice(['circle', 'rect', 'point'], size=nb_obstacles)

    for t in types:
        cx = rng.uniform(x_min, x_max)
        cy = rng.uniform(y_min, y_max)

        if t == 'circle':
            r = rng.uniform(0.3, 0.9)
            circles.append((float(cx), float(cy), float(r)))
        elif t == 'rect':
            w = rng.uniform(0.5, 1.5)
            h = rng.uniform(0.5, 1.5)
            rects.append((float(cx - w/2), float(cy - h/2), float(w), float(h)))
        elif t == 'point':
            points.append((float(cx), float(cy)))

    return circles, rects, points, (x_min, x_max, y_min, y_max)


def draw_obstacles(ax, circles, rects, points):
    for (cx, cy, r) in circles:
        ax.add_patch(plt.Circle((cx, cy), r, color='navy', alpha=0.6, zorder=3))
    for (rx, ry, w, h) in rects:
        ax.add_patch(patches.Rectangle((rx, ry), w, h, color='navy', alpha=0.6, zorder=3))
    for (px, py) in points:
        ax.plot(px, py, 's', color='navy', markersize=10, zorder=3)


def draw_zone(ax, bounds):
    x_min, x_max, y_min, y_max = bounds
    ax.add_patch(patches.Rectangle(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        linewidth=1.2, edgecolor='black', facecolor='none', linestyle='--', label='Zone trajet'
    ))


# -------------------------------------------------------
# CALCUL DU POINT LE PLUS PROCHE POUR UN RECTANGLE
# -------------------------------------------------------
def closest_point_on_rect(px, py, rx, ry, w, h):
    cx = np.clip(px, rx, rx + w)
    cy = np.clip(py, ry, ry + h)
    return np.array([cx, cy])


# -------------------------------------------------------
# ÉVITEMENT PAR CONTOURNEMENT (champ tangentiel)
# -------------------------------------------------------
def compute_avoidance_ctrl(robot_state, circles, rects, points,
                            detection_radius, k_avoid=1.5):
    ctrl = np.zeros((2, 1))
    pos = robot_state.flatten()

    obstacle_points = []

    for (cx, cy, r) in circles:
        obs = np.array([cx, cy])
        d = float(np.linalg.norm(pos - obs)) - r
        obstacle_points.append((obs, max(d, 0.01)))

    for (rx, ry, w, h) in rects:
        obs = closest_point_on_rect(pos[0], pos[1], rx, ry, w, h)
        d = float(np.linalg.norm(pos - obs))
        obstacle_points.append((obs, max(d, 0.01)))

    for (px, py) in points:
        obs = np.array([px, py])
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
            if np.dot(tangent_ccw, goal_dir) >= np.dot(tangent_cw, goal_dir):
                tangent = tangent_ccw
            else:
                tangent = tangent_cw

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
# RÉFÉRENCE ET OBSTACLES
# -------------------------------------------------------
reference = np.array([[8.0], [8.0]])
DETECTION_RADIUS = 2.0

circle_obstacles, rect_obstacles, point_obstacles, zone_bounds = generate_obstacles(
    fleet, reference, nb_obstacles=4, seed=None
)

# -------------------------------------------------------
# GRAPHE DE COMMUNICATION
# -------------------------------------------------------
communicationGraph = Graph.Graph(nbOfRobots)
communicationGraph.adjacencyMatrix = np.ones((6, 6))
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

# -------------------------------------------------------
# INITIALISATION FIGURE — compatible Spyder
# -------------------------------------------------------
plt.close('all')
plt.ion()
fig, ax = plt.subplots(figsize=(7, 7))
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
        fleet.robot[r].ctrl += kr * (reference - fleet.robot[r].state)

        # 3) Évitement obstacles
        fleet.robot[r].ctrl += compute_avoidance_ctrl(
            fleet.robot[r].state,
            circle_obstacles, rect_obstacles, point_obstacles,
            DETECTION_RADIUS, k_avoid=k_obstacle_avoid
        )

    simulation.addDataFromFleet(fleet)
    fleet.integrateMotion(Te)

    # --- ANIMATION ---
    ax.clear()
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_title(f't = {t:.2f} s  |  [p] pour arrêter')
    ax.grid(True, linestyle='--', alpha=0.4)

    draw_zone(ax, zone_bounds)
    draw_obstacles(ax, circle_obstacles, rect_obstacles, point_obstacles)

    ax.plot(float(reference[0, 0]), float(reference[1, 0]),
            '*', color='red', markersize=15, label='Référence')

    for i in range(nbOfRobots):
        x = float(fleet.robot[i].state[0, 0])
        y = float(fleet.robot[i].state[1, 0])
        ax.plot(x, y, 'o', markersize=10, label=f'R{i}')

    ax.legend(loc='upper left', fontsize=7)
    fig.canvas.draw_idle()
    plt.pause(0.001)