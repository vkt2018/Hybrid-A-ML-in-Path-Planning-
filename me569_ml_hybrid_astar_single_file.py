"""
Single-file Hybrid A* Path Planning Demo for ME 569.
Run: python me569_hybrid_astar_single_file.py
"""
import heapq
import math
from math import cos, sin, tan, pi
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

try:
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except Exception:
    RandomForestClassifier = None
    SKLEARN_AVAILABLE = False


def angle_mod(x):
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def rot_mat_2d(angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[c, -s], [s, c]])


# =========================
# DP HEURISTIC
# =========================



class DPNode:

    def __init__(self, x, y, cost, parent_index):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent_index = parent_index

    def __str__(self):
        return str(self.x) + "," + str(self.y) + "," + str(
            self.cost) + "," + str(self.parent_index)


def calc_final_path(goal_node, closed_node_set, resolution):
    # generate final course
    rx, ry = [goal_node.x * resolution], [goal_node.y * resolution]
    parent_index = goal_node.parent_index
    while parent_index != -1:
        n = closed_node_set[parent_index]
        rx.append(n.x * resolution)
        ry.append(n.y * resolution)
        parent_index = n.parent_index

    return rx, ry


def calc_distance_heuristic(gx, gy, ox, oy, resolution, rr):
    """
    gx: goal x position [m]
    gx: goal x position [m]
    ox: x position list of Obstacles [m]
    oy: y position list of Obstacles [m]
    resolution: grid resolution [m]
    rr: robot radius[m]
    """

    goal_node = DPNode(round(gx / resolution), round(gy / resolution), 0.0, -1)
    ox = [iox / resolution for iox in ox]
    oy = [ioy / resolution for ioy in oy]

    obstacle_map, min_x, min_y, max_x, max_y, x_w, y_w = calc_obstacle_map(
        ox, oy, resolution, rr)

    motion = get_motion_model()

    open_set, closed_set = dict(), dict()
    open_set[calc_grid_index(goal_node, x_w, min_x, min_y)] = goal_node
    priority_queue = [(0, calc_grid_index(goal_node, x_w, min_x, min_y))]

    while True:
        if not priority_queue:
            break
        cost, c_id = heapq.heappop(priority_queue)
        if c_id in open_set:
            current = open_set[c_id]
            closed_set[c_id] = current
            open_set.pop(c_id)
        else:
            continue

        # show graph
        if show_animation:  # pragma: no cover
            plt.plot(current.x * resolution, current.y * resolution, "xc")
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            if len(closed_set.keys()) % 10 == 0:
                plt.pause(0.001)

        # Remove the item from the open set

        # expand search grid based on motion model
        for i, _ in enumerate(motion):
            node = DPNode(current.x + motion[i][0],
                        current.y + motion[i][1],
                        current.cost + motion[i][2], c_id)
            n_id = calc_grid_index(node, x_w, min_x, min_y)

            if n_id in closed_set:
                continue

            if not verify_node(node, obstacle_map, min_x, min_y, max_x, max_y):
                continue

            if n_id not in open_set:
                open_set[n_id] = node  # Discover a new node
                heapq.heappush(
                    priority_queue,
                    (node.cost, calc_grid_index(node, x_w, min_x, min_y)))
            else:
                if open_set[n_id].cost >= node.cost:
                    # This path is the best until now. record it!
                    open_set[n_id] = node
                    heapq.heappush(
                        priority_queue,
                        (node.cost, calc_grid_index(node, x_w, min_x, min_y)))

    return closed_set


def verify_node(node, obstacle_map, min_x, min_y, max_x, max_y):
    if node.x < min_x:
        return False
    elif node.y < min_y:
        return False
    elif node.x >= max_x:
        return False
    elif node.y >= max_y:
        return False

    if obstacle_map[node.x][node.y]:
        return False

    return True


def calc_obstacle_map(ox, oy, resolution, vr):
    min_x = round(min(ox))
    min_y = round(min(oy))
    max_x = round(max(ox))
    max_y = round(max(oy))

    x_width = round(max_x - min_x)
    y_width = round(max_y - min_y)

    # obstacle map generation
    obstacle_map = [[False for _ in range(y_width)] for _ in range(x_width)]
    for ix in range(x_width):
        x = ix + min_x
        for iy in range(y_width):
            y = iy + min_y
            #  print(x, y)
            for iox, ioy in zip(ox, oy):
                d = math.hypot(iox - x, ioy - y)
                if d <= vr / resolution:
                    obstacle_map[ix][iy] = True
                    break

    return obstacle_map, min_x, min_y, max_x, max_y, x_width, y_width


def calc_grid_index(node, x_width, x_min, y_min):
    return (node.y - y_min) * x_width + (node.x - x_min)


def get_motion_model():
    # dx, dy, cost
    motion = [[1, 0, 1],
              [0, 1, 1],
              [-1, 0, 1],
              [0, -1, 1],
              [-1, -1, math.sqrt(2)],
              [-1, 1, math.sqrt(2)],
              [1, -1, math.sqrt(2)],
              [1, 1, math.sqrt(2)]]

    return motion

# =========================
# REEDS-SHEPP PATHS
# =========================
class RSPath:
    """
    Path data container
    """

    def __init__(self):
        # course segment length  (negative value is backward segment)
        self.lengths = []
        # course segment type char ("S": straight, "L": left, "R": right)
        self.ctypes = []
        self.L = 0.0  # Total lengths of the path
        self.x = []  # x positions
        self.y = []  # y positions
        self.yaw = []  # orientations [rad]
        self.directions = []  # directions (1:forward, -1:backward)


def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc="r", ec="k"):
    if isinstance(x, list):
        for (ix, iy, iyaw) in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw)
    else:
        plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw), fc=fc,
                  ec=ec, head_width=width, head_length=width)
        plt.plot(x, y)


def pi_2_pi(x):
    return angle_mod(x)

def mod2pi(x):
    # Be consistent with fmod in cplusplus here.
    v = np.mod(x, np.copysign(2.0 * math.pi, x))
    if v < -math.pi:
        v += 2.0 * math.pi
    else:
        if v > math.pi:
            v -= 2.0 * math.pi
    return v

def set_path(paths, lengths, ctypes, step_size):
    path = RSPath()
    path.ctypes = ctypes
    path.lengths = lengths
    path.L = sum(np.abs(lengths))

    # check same path exist
    for i_path in paths:
        type_is_same = (i_path.ctypes == path.ctypes)
        length_is_close = (sum(np.abs(i_path.lengths)) - path.L) <= step_size
        if type_is_same and length_is_close:
            return paths  # same path found, so do not insert path

    # check path is long enough
    if path.L <= step_size:
        return paths  # too short, so do not insert path

    paths.append(path)
    return paths


def polar(x, y):
    r = math.hypot(x, y)
    theta = math.atan2(y, x)
    return r, theta


def left_straight_left(x, y, phi):
    u, t = polar(x - math.sin(phi), y - 1.0 + math.cos(phi))
    if 0.0 <= t <= math.pi:
        v = mod2pi(phi - t)
        if 0.0 <= v <= math.pi:
            return True, [t, u, v], ['L', 'S', 'L']

    return False, [], []


def left_straight_right(x, y, phi):
    u1, t1 = polar(x + math.sin(phi), y - 1.0 - math.cos(phi))
    u1 = u1 ** 2
    if u1 >= 4.0:
        u = math.sqrt(u1 - 4.0)
        theta = math.atan2(2.0, u)
        t = mod2pi(t1 + theta)
        v = mod2pi(t - phi)

        if (t >= 0.0) and (v >= 0.0):
            return True, [t, u, v], ['L', 'S', 'R']

    return False, [], []


def left_x_right_x_left(x, y, phi):
    zeta = x - math.sin(phi)
    eeta = y - 1 + math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 <= 4.0:
        A = math.acos(0.25 * u1)
        t = mod2pi(A + theta + math.pi/2)
        u = mod2pi(math.pi - 2 * A)
        v = mod2pi(phi - t - u)
        return True, [t, -u, v], ['L', 'R', 'L']

    return False, [], []


def left_x_right_left(x, y, phi):
    zeta = x - math.sin(phi)
    eeta = y - 1 + math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 <= 4.0:
        A = math.acos(0.25 * u1)
        t = mod2pi(A + theta + math.pi/2)
        u = mod2pi(math.pi - 2*A)
        v = mod2pi(-phi + t + u)
        return True, [t, -u, -v], ['L', 'R', 'L']

    return False, [], []


def left_right_x_left(x, y, phi):
    zeta = x - math.sin(phi)
    eeta = y - 1 + math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 <= 4.0:
        u = math.acos(1 - u1**2 * 0.125)
        A = math.asin(2 * math.sin(u) / u1)
        t = mod2pi(-A + theta + math.pi/2)
        v = mod2pi(t - u - phi)
        return True, [t, u, -v], ['L', 'R', 'L']

    return False, [], []


def left_right_x_left_right(x, y, phi):
    zeta = x + math.sin(phi)
    eeta = y - 1 - math.cos(phi)
    u1, theta = polar(zeta, eeta)

    # Solutions refering to (2 < u1 <= 4) are considered sub-optimal in paper
    # Solutions do not exist for u1 > 4
    if u1 <= 2:
        A = math.acos((u1 + 2) * 0.25)
        t = mod2pi(theta + A + math.pi/2)
        u = mod2pi(A)
        v = mod2pi(phi - t + 2*u)
        if ((t >= 0) and (u >= 0) and (v >= 0)):
            return True, [t, u, -u, -v], ['L', 'R', 'L', 'R']

    return False, [], []


def left_x_right_left_x_right(x, y, phi):
    zeta = x + math.sin(phi)
    eeta = y - 1 - math.cos(phi)
    u1, theta = polar(zeta, eeta)
    u2 = (20 - u1**2) / 16

    if (0 <= u2 <= 1):
        u = math.acos(u2)
        A = math.asin(2 * math.sin(u) / u1)
        t = mod2pi(theta + A + math.pi/2)
        v = mod2pi(t - phi)
        if (t >= 0) and (v >= 0):
            return True, [t, -u, -u, v], ['L', 'R', 'L', 'R']

    return False, [], []


def left_x_right90_straight_left(x, y, phi):
    zeta = x - math.sin(phi)
    eeta = y - 1 + math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 >= 2.0:
        u = math.sqrt(u1**2 - 4) - 2
        A = math.atan2(2, math.sqrt(u1**2 - 4))
        t = mod2pi(theta + A + math.pi/2)
        v = mod2pi(t - phi + math.pi/2)
        if (t >= 0) and (v >= 0):
           return True, [t, -math.pi/2, -u, -v], ['L', 'R', 'S', 'L']

    return False, [], []


def left_straight_right90_x_left(x, y, phi):
    zeta = x - math.sin(phi)
    eeta = y - 1 + math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 >= 2.0:
        u = math.sqrt(u1**2 - 4) - 2
        A = math.atan2(math.sqrt(u1**2 - 4), 2)
        t = mod2pi(theta - A + math.pi/2)
        v = mod2pi(t - phi - math.pi/2)
        if (t >= 0) and (v >= 0):
            return True, [t, u, math.pi/2, -v], ['L', 'S', 'R', 'L']

    return False, [], []


def left_x_right90_straight_right(x, y, phi):
    zeta = x + math.sin(phi)
    eeta = y - 1 - math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 >= 2.0:
        t = mod2pi(theta + math.pi/2)
        u = u1 - 2
        v = mod2pi(phi - t - math.pi/2)
        if (t >= 0) and (v >= 0):
            return True, [t, -math.pi/2, -u, -v], ['L', 'R', 'S', 'R']

    return False, [], []


def left_straight_left90_x_right(x, y, phi):
    zeta = x + math.sin(phi)
    eeta = y - 1 - math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 >= 2.0:
        t = mod2pi(theta)
        u = u1 - 2
        v = mod2pi(phi - t - math.pi/2)
        if (t >= 0) and (v >= 0):
            return True, [t, u, math.pi/2, -v], ['L', 'S', 'L', 'R']

    return False, [], []


def left_x_right90_straight_left90_x_right(x, y, phi):
    zeta = x + math.sin(phi)
    eeta = y - 1 - math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 >= 4.0:
        u = math.sqrt(u1**2 - 4) - 4
        A = math.atan2(2, math.sqrt(u1**2 - 4))
        t = mod2pi(theta + A + math.pi/2)
        v = mod2pi(t - phi)
        if (t >= 0) and (v >= 0):
            return True, [t, -math.pi/2, -u, -math.pi/2, v], ['L', 'R', 'S', 'L', 'R']

    return False, [], []


def timeflip(travel_distances):
    return [-x for x in travel_distances]


def reflect(steering_directions):
    def switch_dir(dirn):
        if dirn == 'L':
            return 'R'
        elif dirn == 'R':
            return 'L'
        else:
            return 'S'
    return[switch_dir(dirn) for dirn in steering_directions]


def generate_path(q0, q1, max_curvature, step_size):
    dx = q1[0] - q0[0]
    dy = q1[1] - q0[1]
    dth = q1[2] - q0[2]
    c = math.cos(q0[2])
    s = math.sin(q0[2])
    x = (c * dx + s * dy) * max_curvature
    y = (-s * dx + c * dy) * max_curvature
    step_size *= max_curvature

    paths = []
    path_functions = [left_straight_left, left_straight_right,                          # CSC
                      left_x_right_x_left, left_x_right_left, left_right_x_left,        # CCC
                      left_right_x_left_right, left_x_right_left_x_right,               # CCCC
                      left_x_right90_straight_left, left_x_right90_straight_right,      # CCSC
                      left_straight_right90_x_left, left_straight_left90_x_right,       # CSCC
                      left_x_right90_straight_left90_x_right]                           # CCSCC

    for path_func in path_functions:
        flag, travel_distances, steering_dirns = path_func(x, y, dth)
        if flag:
            for distance in travel_distances:
                if (0.1*sum([abs(d) for d in travel_distances]) < abs(distance) < step_size):
                    print("Step size too large for Reeds-Shepp paths.")
                    return []
            paths = set_path(paths, travel_distances, steering_dirns, step_size)

        flag, travel_distances, steering_dirns = path_func(-x, y, -dth)
        if flag:
            for distance in travel_distances:
                if (0.1*sum([abs(d) for d in travel_distances]) < abs(distance) < step_size):
                    print("Step size too large for Reeds-Shepp paths.")
                    return []
            travel_distances = timeflip(travel_distances)
            paths = set_path(paths, travel_distances, steering_dirns, step_size)

        flag, travel_distances, steering_dirns = path_func(x, -y, -dth)
        if flag:
            for distance in travel_distances:
                if (0.1*sum([abs(d) for d in travel_distances]) < abs(distance) < step_size):
                    print("Step size too large for Reeds-Shepp paths.")
                    return []
            steering_dirns = reflect(steering_dirns)
            paths = set_path(paths, travel_distances, steering_dirns, step_size)

        flag, travel_distances, steering_dirns = path_func(-x, -y, dth)
        if flag:
            for distance in travel_distances:
                if (0.1*sum([abs(d) for d in travel_distances]) < abs(distance) < step_size):
                    print("Step size too large for Reeds-Shepp paths.")
                    return []
            travel_distances = timeflip(travel_distances)
            steering_dirns = reflect(steering_dirns)
            paths = set_path(paths, travel_distances, steering_dirns, step_size)

    return paths


def calc_interpolate_dists_list(lengths, step_size):
    interpolate_dists_list = []
    for length in lengths:
        d_dist = step_size if length >= 0.0 else -step_size
        interp_dists = np.arange(0.0, length, d_dist)
        interp_dists = np.append(interp_dists, length)
        interpolate_dists_list.append(interp_dists)

    return interpolate_dists_list


def generate_local_course(lengths, modes, max_curvature, step_size):
    interpolate_dists_list = calc_interpolate_dists_list(lengths, step_size * max_curvature)

    origin_x, origin_y, origin_yaw = 0.0, 0.0, 0.0

    xs, ys, yaws, directions = [], [], [], []
    for (interp_dists, mode, length) in zip(interpolate_dists_list, modes,
                                            lengths):

        for dist in interp_dists:
            x, y, yaw, direction = interpolate(dist, length, mode,
                                               max_curvature, origin_x,
                                               origin_y, origin_yaw)
            xs.append(x)
            ys.append(y)
            yaws.append(yaw)
            directions.append(direction)
        origin_x = xs[-1]
        origin_y = ys[-1]
        origin_yaw = yaws[-1]

    return xs, ys, yaws, directions


def interpolate(dist, length, mode, max_curvature, origin_x, origin_y,
                origin_yaw):
    if mode == "S":
        x = origin_x + dist / max_curvature * math.cos(origin_yaw)
        y = origin_y + dist / max_curvature * math.sin(origin_yaw)
        yaw = origin_yaw
    else:  # curve
        ldx = math.sin(dist) / max_curvature
        ldy = 0.0
        yaw = None
        if mode == "L":  # left turn
            ldy = (1.0 - math.cos(dist)) / max_curvature
            yaw = origin_yaw + dist
        elif mode == "R":  # right turn
            ldy = (1.0 - math.cos(dist)) / -max_curvature
            yaw = origin_yaw - dist
        gdx = math.cos(-origin_yaw) * ldx + math.sin(-origin_yaw) * ldy
        gdy = -math.sin(-origin_yaw) * ldx + math.cos(-origin_yaw) * ldy
        x = origin_x + gdx
        y = origin_y + gdy

    return x, y, yaw, 1 if length > 0.0 else -1


def calc_paths(sx, sy, syaw, gx, gy, gyaw, maxc, step_size):
    q0 = [sx, sy, syaw]
    q1 = [gx, gy, gyaw]

    paths = generate_path(q0, q1, maxc, step_size)
    for path in paths:
        xs, ys, yaws, directions = generate_local_course(path.lengths,
                                                         path.ctypes, maxc,
                                                         step_size)

        # convert global coordinate
        path.x = [math.cos(-q0[2]) * ix + math.sin(-q0[2]) * iy + q0[0] for
                  (ix, iy) in zip(xs, ys)]
        path.y = [-math.sin(-q0[2]) * ix + math.cos(-q0[2]) * iy + q0[1] for
                  (ix, iy) in zip(xs, ys)]
        path.yaw = [pi_2_pi(yaw + q0[2]) for yaw in yaws]
        path.directions = directions
        path.lengths = [length / maxc for length in path.lengths]
        path.L = path.L / maxc

    return paths


def reeds_shepp_path_planning(sx, sy, syaw, gx, gy, gyaw, maxc, step_size=0.2):
    paths = calc_paths(sx, sy, syaw, gx, gy, gyaw, maxc, step_size)
    if not paths:
        return None, None, None, None, None  # could not generate any path

    # search minimum cost path
    best_path_index = paths.index(min(paths, key=lambda p: abs(p.L)))
    b_path = paths[best_path_index]

    return b_path.x, b_path.y, b_path.yaw, b_path.ctypes, b_path.lengths


def main():
    print("Reeds Shepp path planner sample start!!")

    start_x = -1.0  # [m]
    start_y = -4.0  # [m]
    start_yaw = np.deg2rad(-20.0)  # [rad]

    end_x = 5.0  # [m]
    end_y = 5.0  # [m]
    end_yaw = np.deg2rad(25.0)  # [rad]

    curvature = 0.1
    step_size = 0.05

    xs, ys, yaws, modes, lengths = reeds_shepp_path_planning(start_x, start_y,
                                                             start_yaw, end_x,
                                                             end_y, end_yaw,
                                                             curvature,
                                                             step_size)

    if not xs:
        assert False, "No path"

    if show_animation:  # pragma: no cover
        plt.cla()
        plt.plot(xs, ys, label="final course " + str(modes))
        print(f"{lengths=}")

        # plotting
        plot_arrow(start_x, start_y, start_yaw)
        plot_arrow(end_x, end_y, end_yaw)

        plt.legend()
        plt.grid(True)
        plt.axis("equal")
        plt.show()



# =========================
# CAR MODEL
# =========================
WB = 3.0  # rear to front wheel
W = 2.0  # width of car
LF = 3.3  # distance from rear to vehicle front end
LB = 1.0  # distance from rear to vehicle back end
MAX_STEER = 0.6  # [rad] maximum steering angle

BUBBLE_DIST = (LF - LB) / 2.0  # distance from rear to center of vehicle.
BUBBLE_R = np.hypot((LF + LB) / 2.0, W / 2.0)  # bubble radius

# vehicle rectangle vertices
VRX = [LF, LF, -LB, -LB, LF]
VRY = [W / 2, -W / 2, -W / 2, W / 2, W / 2]


def check_car_collision(x_list, y_list, yaw_list, ox, oy, kd_tree):
    for i_x, i_y, i_yaw in zip(x_list, y_list, yaw_list):
        cx = i_x + BUBBLE_DIST * cos(i_yaw)
        cy = i_y + BUBBLE_DIST * sin(i_yaw)

        ids = kd_tree.query_ball_point([cx, cy], BUBBLE_R)

        if not ids:
            continue

        if not rectangle_check(i_x, i_y, i_yaw,
                               [ox[i] for i in ids], [oy[i] for i in ids]):
            return False  # collision

    return True  # no collision


def rectangle_check(x, y, yaw, ox, oy):
    # transform obstacles to base link frame
    rot = rot_mat_2d(yaw)
    for iox, ioy in zip(ox, oy):
        tx = iox - x
        ty = ioy - y
        converted_xy = np.stack([tx, ty]).T @ rot
        rx, ry = converted_xy[0], converted_xy[1]

        if not (rx > LF or rx < -LB or ry > W / 2.0 or ry < -W / 2.0):
            return False  # collision

    return True  # no collision


def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc="r", ec="k"):
    """Plot arrow."""
    if not isinstance(x, float):
        for (i_x, i_y, i_yaw) in zip(x, y, yaw):
            plot_arrow(i_x, i_y, i_yaw)
    else:
        plt.arrow(x, y, length * cos(yaw), length * sin(yaw),
                  fc=fc, ec=ec, head_width=width, head_length=width, alpha=0.4)


def plot_car(x, y, yaw):
    car_color = '-k'
    c, s = cos(yaw), sin(yaw)
    rot = rot_mat_2d(-yaw)
    car_outline_x, car_outline_y = [], []
    for rx, ry in zip(VRX, VRY):
        converted_xy = np.stack([rx, ry]).T @ rot
        car_outline_x.append(converted_xy[0]+x)
        car_outline_y.append(converted_xy[1]+y)

    arrow_x, arrow_y, arrow_yaw = c * 1.5 + x, s * 1.5 + y, yaw
    plot_arrow(arrow_x, arrow_y, arrow_yaw)

    plt.plot(car_outline_x, car_outline_y, car_color)


def pi_2_pi(angle):
    return (angle + pi) % (2 * pi) - pi


def move(x, y, yaw, distance, steer, L=WB):
    x += distance * cos(yaw)
    y += distance * sin(yaw)
    yaw = pi_2_pi(yaw + distance * tan(steer) / L)  # distance/2

    return x, y, yaw


def main():
    x, y, yaw = 0., 0., 1.
    plt.axis('equal')
    plot_car(x, y, yaw)
    plt.show()



# =========================
# HYBRID A STAR
# =========================
XY_GRID_RESOLUTION = 2.0  # [m]
YAW_GRID_RESOLUTION = np.deg2rad(15.0)  # [rad]
MOTION_RESOLUTION = 0.1  # [m] path interpolate resolution
N_STEER = 20  # number of steer command

SB_COST = 100.0  # switch back penalty cost
BACK_COST = 5.0  # backward penalty cost
STEER_CHANGE_COST = 5.0  # steer angle change penalty cost
STEER_COST = 1.0  # steer angle change penalty cost
H_COST = 5.0  # Heuristic cost

show_animation = False


class Node:

    def __init__(self, x_ind, y_ind, yaw_ind, direction,
                 x_list, y_list, yaw_list, directions,
                 steer=0.0, parent_index=None, cost=None):
        self.x_index = x_ind
        self.y_index = y_ind
        self.yaw_index = yaw_ind
        self.direction = direction
        self.x_list = x_list
        self.y_list = y_list
        self.yaw_list = yaw_list
        self.directions = directions
        self.steer = steer
        self.parent_index = parent_index
        self.cost = cost


class Path:

    def __init__(self, x_list, y_list, yaw_list, direction_list, cost):
        self.x_list = x_list
        self.y_list = y_list
        self.yaw_list = yaw_list
        self.direction_list = direction_list
        self.cost = cost


class Config:

    def __init__(self, ox, oy, xy_resolution, yaw_resolution):
        min_x_m = min(ox)
        min_y_m = min(oy)
        max_x_m = max(ox)
        max_y_m = max(oy)

        ox.append(min_x_m)
        oy.append(min_y_m)
        ox.append(max_x_m)
        oy.append(max_y_m)

        self.min_x = round(min_x_m / xy_resolution)
        self.min_y = round(min_y_m / xy_resolution)
        self.max_x = round(max_x_m / xy_resolution)
        self.max_y = round(max_y_m / xy_resolution)

        self.x_w = round(self.max_x - self.min_x)
        self.y_w = round(self.max_y - self.min_y)

        self.min_yaw = round(- math.pi / yaw_resolution) - 1
        self.max_yaw = round(math.pi / yaw_resolution)
        self.yaw_w = round(self.max_yaw - self.min_yaw)


def calc_motion_inputs():
    for steer in np.concatenate((np.linspace(-MAX_STEER, MAX_STEER,
                                             N_STEER), [0.0])):
        for d in [1, -1]:
            yield [float(steer), int(d)]


def calc_motion_inputs_list():
    return list(calc_motion_inputs())


ACTION_SPACE = calc_motion_inputs_list()


def get_neighbors(current, config, ox, oy, kd_tree, action_inputs=None):
    if action_inputs is None:
        action_inputs = ACTION_SPACE
    for steer, d in action_inputs:
        node = calc_next_node(current, steer, d, config, ox, oy, kd_tree)
        if node and verify_index(node, config):
            yield node


def calc_next_node(current, steer, direction, config, ox, oy, kd_tree):
    x, y, yaw = current.x_list[-1], current.y_list[-1], current.yaw_list[-1]

    arc_l = XY_GRID_RESOLUTION * 1.5
    x_list, y_list, yaw_list, direction_list = [], [], [], []
    for _ in np.arange(0, arc_l, MOTION_RESOLUTION):
        x, y, yaw = move(x, y, yaw, MOTION_RESOLUTION * direction, steer)
        x_list.append(x)
        y_list.append(y)
        yaw_list.append(yaw)
        direction_list.append(direction == 1)

    if not check_car_collision(x_list, y_list, yaw_list, ox, oy, kd_tree):
        return None

    d = direction == 1
    x_ind = round(x / XY_GRID_RESOLUTION)
    y_ind = round(y / XY_GRID_RESOLUTION)
    yaw_ind = round(yaw / YAW_GRID_RESOLUTION)

    added_cost = 0.0

    if d != current.direction:
        added_cost += SB_COST

    # steer penalty
    added_cost += STEER_COST * abs(steer)

    # steer change penalty
    added_cost += STEER_CHANGE_COST * abs(current.steer - steer)

    cost = current.cost + added_cost + arc_l

    node = Node(x_ind, y_ind, yaw_ind, d, x_list,
                y_list, yaw_list, direction_list,
                parent_index=calc_index(current, config),
                cost=cost, steer=steer)

    return node


def is_same_grid(n1, n2):
    if n1.x_index == n2.x_index \
            and n1.y_index == n2.y_index \
            and n1.yaw_index == n2.yaw_index:
        return True
    return False


def analytic_expansion(current, goal, ox, oy, kd_tree):
    start_x = current.x_list[-1]
    start_y = current.y_list[-1]
    start_yaw = current.yaw_list[-1]

    goal_x = goal.x_list[-1]
    goal_y = goal.y_list[-1]
    goal_yaw = goal.yaw_list[-1]

    max_curvature = math.tan(MAX_STEER) / WB
    paths = calc_paths(start_x, start_y, start_yaw,
                          goal_x, goal_y, goal_yaw,
                          max_curvature, step_size=MOTION_RESOLUTION)

    if not paths:
        return None

    best_path, best = None, None

    for path in paths:
        if check_car_collision(path.x, path.y, path.yaw, ox, oy, kd_tree):
            cost = calc_rs_path_cost(path)
            if not best or best > cost:
                best = cost
                best_path = path

    return best_path


def update_node_with_analytic_expansion(current, goal,
                                        c, ox, oy, kd_tree):
    path = analytic_expansion(current, goal, ox, oy, kd_tree)

    if path:
        if show_animation:
            plt.plot(path.x, path.y)
        f_x = path.x[1:]
        f_y = path.y[1:]
        f_yaw = path.yaw[1:]

        f_cost = current.cost + calc_rs_path_cost(path)
        f_parent_index = calc_index(current, c)

        fd = []
        for d in path.directions[1:]:
            fd.append(d >= 0)

        f_steer = 0.0
        f_path = Node(current.x_index, current.y_index, current.yaw_index,
                      current.direction, f_x, f_y, f_yaw, fd,
                      cost=f_cost, parent_index=f_parent_index, steer=f_steer)
        return True, f_path

    return False, None


def calc_rs_path_cost(reed_shepp_path):
    cost = 0.0
    for length in reed_shepp_path.lengths:
        if length >= 0:  # forward
            cost += length
        else:  # back
            cost += abs(length) * BACK_COST

    # switch back penalty
    for i in range(len(reed_shepp_path.lengths) - 1):
        # switch back
        if reed_shepp_path.lengths[i] * reed_shepp_path.lengths[i + 1] < 0.0:
            cost += SB_COST

    # steer penalty
    for course_type in reed_shepp_path.ctypes:
        if course_type != "S":  # curve
            cost += STEER_COST * abs(MAX_STEER)

    # ==steer change penalty
    # calc steer profile
    n_ctypes = len(reed_shepp_path.ctypes)
    u_list = [0.0] * n_ctypes
    for i in range(n_ctypes):
        if reed_shepp_path.ctypes[i] == "R":
            u_list[i] = - MAX_STEER
        elif reed_shepp_path.ctypes[i] == "L":
            u_list[i] = MAX_STEER

    for i in range(len(reed_shepp_path.ctypes) - 1):
        cost += STEER_CHANGE_COST * abs(u_list[i + 1] - u_list[i])

    return cost



def wrap_to_pi(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def make_ml_features(current, goal_node):
    """Feature vector for the learned action selector."""
    x = current.x_list[-1]
    y = current.y_list[-1]
    yaw = current.yaw_list[-1]
    gx = goal_node.x_list[-1]
    gy = goal_node.y_list[-1]
    gyaw = goal_node.yaw_list[-1]

    dx = gx - x
    dy = gy - y
    dist = math.hypot(dx, dy)
    goal_bearing = math.atan2(dy, dx)
    heading_error = wrap_to_pi(goal_bearing - yaw)
    yaw_error = wrap_to_pi(gyaw - yaw)

    # Goal vector in vehicle frame. These two features are useful because
    # they tell the model whether the goal is ahead/behind and left/right.
    local_dx = math.cos(yaw) * dx + math.sin(yaw) * dy
    local_dy = -math.sin(yaw) * dx + math.cos(yaw) * dy

    return [
        local_dx,
        local_dy,
        dist,
        math.sin(heading_error),
        math.cos(heading_error),
        math.sin(yaw_error),
        math.cos(yaw_error),
        current.steer,
        1.0 if current.direction else -1.0,
    ]


def make_temp_node(x, y, yaw, config):
    return Node(round(x / XY_GRID_RESOLUTION),
                round(y / XY_GRID_RESOLUTION),
                round(yaw / YAW_GRID_RESOLUTION),
                True, [x], [y], [yaw], [True], cost=0.0, steer=0.0)


def build_learning_dataset(goal_node, config, ox, oy, kd_tree, h_dp,
                           n_samples=350, seed=4):
    """Generate supervised data by asking the baseline cost function which
    motion primitive is best from randomly sampled valid states.

    This keeps the project self-contained: the labels come from Hybrid A*'s
    own one-step lookahead cost, and the Random Forest learns to rank actions.
    """
    rng = np.random.default_rng(seed)
    X, y = [], []

    min_x = config.min_x * XY_GRID_RESOLUTION + 2.0
    max_x = config.max_x * XY_GRID_RESOLUTION - 2.0
    min_y = config.min_y * XY_GRID_RESOLUTION + 2.0
    max_y = config.max_y * XY_GRID_RESOLUTION - 2.0

    attempts = 0
    max_attempts = n_samples * 30
    while len(X) < n_samples and attempts < max_attempts:
        attempts += 1
        x = float(rng.uniform(min_x, max_x))
        y0 = float(rng.uniform(min_y, max_y))
        yaw = float(rng.uniform(-math.pi, math.pi))
        current = make_temp_node(x, y0, yaw, config)

        if not check_car_collision([x], [y0], [yaw], ox, oy, kd_tree):
            continue

        best_action_index = None
        best_score = float("inf")
        for action_index, (steer, direction) in enumerate(ACTION_SPACE):
            node = calc_next_node(current, steer, direction, config, ox, oy, kd_tree)
            if node is None or not verify_index(node, config):
                continue
            score = calc_cost(node, h_dp, config)
            if score < best_score:
                best_score = score
                best_action_index = action_index

        if best_action_index is None:
            continue

        X.append(make_ml_features(current, goal_node))
        y.append(best_action_index)

    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)


def train_action_model(goal, ox, oy, xy_resolution, yaw_resolution, n_samples=350):
    """Train a Random Forest action selector for the current map and goal."""
    if not SKLEARN_AVAILABLE:
        print("scikit-learn not found. Install with: pip install scikit-learn")
        return None, 0, 0.0

    tox, toy = ox[:], oy[:]
    kd_tree = cKDTree(np.vstack((tox, toy)).T)
    config = Config(tox, toy, xy_resolution, yaw_resolution)
    goal_node = Node(round(goal[0] / xy_resolution),
                     round(goal[1] / xy_resolution),
                     round(goal[2] / yaw_resolution), True,
                     [goal[0]], [goal[1]], [goal[2]], [True])

    h_dp = calc_distance_heuristic(goal_node.x_list[-1], goal_node.y_list[-1],
                                   ox, oy, xy_resolution, BUBBLE_R)

    t0 = time.perf_counter()
    X, labels = build_learning_dataset(goal_node, config, ox, oy, kd_tree, h_dp,
                                       n_samples=n_samples)
    model = RandomForestClassifier(
        n_estimators=80,
        max_depth=10,
        min_samples_leaf=2,
        random_state=0,
        n_jobs=-1,
    )
    model.fit(X, labels)
    training_time = time.perf_counter() - t0
    return model, len(labels), training_time


def ml_ranked_actions(current, goal_node, model, top_k=6):
    """Return a small set of likely useful actions for ML-guided Hybrid A*."""
    if model is None:
        return ACTION_SPACE

    features = np.asarray([make_ml_features(current, goal_node)], dtype=float)
    probabilities = model.predict_proba(features)[0]
    classes = model.classes_
    order = np.argsort(probabilities)[::-1]

    selected = []
    selected_indices = set()
    for idx in order[:top_k]:
        action_index = int(classes[idx])
        selected.append(ACTION_SPACE[action_index])
        selected_indices.add(action_index)

    # Safety fallback: always include straight forward and straight reverse.
    # This prevents the learned planner from getting stuck if the model is overconfident.
    for i, action in enumerate(ACTION_SPACE):
        steer, direction = action
        if abs(steer) < 1e-9 and i not in selected_indices:
            selected.append(action)
            selected_indices.add(i)

    return selected

def hybrid_a_star_planning(start, goal, ox, oy, xy_resolution, yaw_resolution, mode="baseline", model=None, top_k=6):
    """
    start: start node
    goal: goal node
    ox: x position list of Obstacles [m]
    oy: y position list of Obstacles [m]
    xy_resolution: grid resolution [m]
    yaw_resolution: yaw angle resolution [rad]
    """

    start[2], goal[2] = pi_2_pi(start[2]), pi_2_pi(goal[2])
    tox, toy = ox[:], oy[:]

    obstacle_kd_tree = cKDTree(np.vstack((tox, toy)).T)

    config = Config(tox, toy, xy_resolution, yaw_resolution)

    start_node = Node(round(start[0] / xy_resolution),
                      round(start[1] / xy_resolution),
                      round(start[2] / yaw_resolution), True,
                      [start[0]], [start[1]], [start[2]], [True], cost=0)
    goal_node = Node(round(goal[0] / xy_resolution),
                     round(goal[1] / xy_resolution),
                     round(goal[2] / yaw_resolution), True,
                     [goal[0]], [goal[1]], [goal[2]], [True])

    openList, closedList = {}, {}

    h_dp = calc_distance_heuristic(
        goal_node.x_list[-1], goal_node.y_list[-1],
        ox, oy, xy_resolution, BUBBLE_R)

    pq = []
    openList[calc_index(start_node, config)] = start_node
    heapq.heappush(pq, (calc_cost(start_node, h_dp, config),
                        calc_index(start_node, config)))
    final_path = None
    nodes_expanded = 0

    while True:
        if not openList:
            print("Error: Cannot find path, No open set")
            return Path([], [], [], [], 0)

        cost, c_id = heapq.heappop(pq)
        if c_id in openList:
            current = openList.pop(c_id)
            closedList[c_id] = current
            nodes_expanded += 1
        else:
            continue

        if show_animation:  # pragma: no cover
            plt.plot(current.x_list[-1], current.y_list[-1], "xc")
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            if len(closedList.keys()) % 10 == 0:
                plt.pause(0.001)

        is_updated, final_path = update_node_with_analytic_expansion(
            current, goal_node, config, ox, oy, obstacle_kd_tree)

        if is_updated:
            print("path found")
            break

        if mode == "ml":
            action_inputs = ml_ranked_actions(current, goal_node, model, top_k=top_k)
        else:
            action_inputs = ACTION_SPACE

        for neighbor in get_neighbors(current, config, ox, oy,
                                      obstacle_kd_tree, action_inputs=action_inputs):
            neighbor_index = calc_index(neighbor, config)
            if neighbor_index in closedList:
                continue
            if neighbor_index not in openList \
                    or openList[neighbor_index].cost > neighbor.cost:
                heapq.heappush(
                    pq, (calc_cost(neighbor, h_dp, config),
                         neighbor_index))
                openList[neighbor_index] = neighbor

    path = get_final_path(closedList, final_path)
    path.nodes_expanded = nodes_expanded
    return path


def calc_cost(n, h_dp, c):
    ind = (n.y_index - c.min_y) * c.x_w + (n.x_index - c.min_x)
    if ind not in h_dp:
        return n.cost + 999999999  # collision cost
    return n.cost + H_COST * h_dp[ind].cost


def get_final_path(closed, goal_node):
    reversed_x, reversed_y, reversed_yaw = \
        list(reversed(goal_node.x_list)), list(reversed(goal_node.y_list)), \
        list(reversed(goal_node.yaw_list))
    direction = list(reversed(goal_node.directions))
    nid = goal_node.parent_index
    final_cost = goal_node.cost

    while nid:
        n = closed[nid]
        reversed_x.extend(list(reversed(n.x_list)))
        reversed_y.extend(list(reversed(n.y_list)))
        reversed_yaw.extend(list(reversed(n.yaw_list)))
        direction.extend(list(reversed(n.directions)))

        nid = n.parent_index

    reversed_x = list(reversed(reversed_x))
    reversed_y = list(reversed(reversed_y))
    reversed_yaw = list(reversed(reversed_yaw))
    direction = list(reversed(direction))

    # adjust first direction
    direction[0] = direction[1]

    path = Path(reversed_x, reversed_y, reversed_yaw, direction, final_cost)

    return path


def verify_index(node, c):
    x_ind, y_ind = node.x_index, node.y_index
    if c.min_x <= x_ind <= c.max_x and c.min_y <= y_ind <= c.max_y:
        return True

    return False


def calc_index(node, c):
    ind = (node.yaw_index - c.min_yaw) * c.x_w * c.y_w + \
          (node.y_index - c.min_y) * c.x_w + (node.x_index - c.min_x)

    if ind <= 0:
        print("Error(calc_index):", ind)

    return ind


def build_demo_map():
    ox, oy = [], []

    for i in range(60):
        ox.append(float(i))
        oy.append(0.0)
    for i in range(60):
        ox.append(60.0)
        oy.append(float(i))
    for i in range(61):
        ox.append(float(i))
        oy.append(60.0)
    for i in range(61):
        ox.append(0.0)
        oy.append(float(i))
    for i in range(40):
        ox.append(20.0)
        oy.append(float(i))
    for i in range(40):
        ox.append(40.0)
        oy.append(60.0 - float(i))

    return ox, oy


def compute_path_metrics(path):
    x = path.x_list
    y = path.y_list
    if len(x) < 2:
        return 0.0, 0
    path_length = sum(math.hypot(x[i] - x[i - 1], y[i] - y[i - 1])
                      for i in range(1, len(x)))
    direction_switches = sum(
        1 for i in range(1, len(path.direction_list))
        if path.direction_list[i] != path.direction_list[i - 1]
    )
    return path_length, direction_switches


def run_planner_case(name, start, goal, ox, oy, mode="baseline", model=None, top_k=6):
    t0 = time.perf_counter()
    path = hybrid_a_star_planning(
        start[:], goal[:], ox, oy,
        XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION,
        mode=mode, model=model, top_k=top_k,
    )
    runtime = time.perf_counter() - t0
    path_length, direction_switches = compute_path_metrics(path)

    metrics = {
        "method": name,
        "runtime": runtime,
        "nodes": getattr(path, "nodes_expanded", 0),
        "cost": path.cost,
        "length": path_length,
        "switches": direction_switches,
    }
    return path, metrics


def print_metrics_table(rows):
    print("\n--- Comparison Metrics ---")
    print(f"{'Method':<18} {'Runtime(s)':>12} {'Nodes':>8} {'Cost':>12} {'Length(m)':>12} {'Switches':>10}")
    print("-" * 76)
    for r in rows:
        print(f"{r['method']:<18} {r['runtime']:>12.4f} {r['nodes']:>8d} "
              f"{r['cost']:>12.4f} {r['length']:>12.4f} {r['switches']:>10d}")


def plot_comparison(ox, oy, start, goal, baseline_path, ml_path):
    plt.figure()
    plt.plot(ox, oy, ".k", label="Obstacles")
    if len(baseline_path.x_list) > 0:
        plt.plot(baseline_path.x_list, baseline_path.y_list, "-r", label="Baseline Hybrid A*")
    if len(ml_path.x_list) > 0:
        plt.plot(ml_path.x_list, ml_path.y_list, "--b", label="ML-guided Hybrid A*")

    plot_arrow(start[0], start[1], start[2], fc="g")
    plot_arrow(goal[0], goal[1], goal[2], fc="b")

    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.title("Baseline vs ML-guided Hybrid A*")
    plt.savefig("hybrid_astar_ml_comparison.png", dpi=200, bbox_inches="tight")
    plt.show()


def main():
    print("Start ME 569 Hybrid A* comparison")

    ox, oy = build_demo_map()
    start = [10.0, 10.0, np.deg2rad(90.0)]
    goal = [50.0, 50.0, np.deg2rad(-90.0)]

    print("start:", start)
    print("goal :", goal)

    print("\nRunning baseline Hybrid A*...")
    baseline_path, baseline_metrics = run_planner_case(
        "Baseline", start, goal, ox, oy, mode="baseline"
    )

    print("\nTraining Random Forest action selector...")
    model, n_train, train_time = train_action_model(
        goal[:], ox, oy, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION,
        n_samples=350,
    )
    print(f"Training samples: {n_train}")
    print(f"Training time: {train_time:.4f} seconds")

    print("\nRunning ML-guided Hybrid A*...")
    ml_path, ml_metrics = run_planner_case(
        "ML-guided", start, goal, ox, oy,
        mode="ml", model=model, top_k=6,
    )

    print_metrics_table([baseline_metrics, ml_metrics])
    plot_comparison(ox, oy, start, goal, baseline_path, ml_path)
    print("\nSaved figure: hybrid_astar_ml_comparison.png")

if __name__ == '__main__':
    main()
