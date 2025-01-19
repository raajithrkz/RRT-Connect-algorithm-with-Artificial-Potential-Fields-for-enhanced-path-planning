"""
This code defines the RRT-Connect class that implements the RRT-Connect
algorithm.

The code imports required libraries, declares some constants and defines required functions for
calculating tasks like field force,steering and costs.

The class RRTConnect initializes the algorithm with dimensions, sampling
strategy, start and end points, maximum samples, step size, obstacles, probability of random
connection.

The bidirectional algorithm incrementally expands trees by adding vertices, endeavors to link them,
and exchanges trees from both directions as needed. It yields the shortest discovered path once either
the designated number of samples is attained or the criteria for random connection probability is met.

The plot() function is used to visualize the path, trees, and obstacles.

Finally, the main part of the code sets the dimensions, obstacles, and other parameters for the
algorithm, runs it, and visualizes the result.

Note that this code requires the 'numpy','matplotlib' libraries.
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

class Utils:
    def __init__(self):
        """
        Initializes the Utils class with default environment settings.
        Loads obstacle parameters from the Env class.
        """
        self.env = Env()

        self.delta = 0.5
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary

    def update_obs(self, obs_cir, obs_bound, obs_rec):
        """
        Updates the obstacles in the environment.

        Args:
            obs_cir (list): Circular obstacles.
            obs_bound (list): Boundary obstacles.
            obs_rec (list): Rectangular obstacles.
        """
        self.obs_circle = obs_cir
        self.obs_boundary = obs_bound
        self.obs_rectangle = obs_rec

    def get_obs_vertex(self):
        """
        Returns a list of vertices for each rectangular obstacle,
        expanding them by the value of `self.delta`.

        Returns:
            obs_list (list): List of vertices for each rectangular obstacle.
        """
        delta = self.delta
        obs_list = []

        for (ox, oy, w, h) in self.obs_rectangle:
            vertex_list = [[ox - delta, oy - delta],
                           [ox + w + delta, oy - delta],
                           [ox + w + delta, oy + h + delta],
                           [ox - delta, oy + h + delta]]
            obs_list.append(vertex_list)

        return obs_list

    def is_intersect_rec(self, start, end, o, d, a, b):
        """
        Checks if a given line intersects a rectangle side.

        Args:
            start (Node): Start point of the line.
            end (Node): End point of the line.
            o (list): Origin of the ray.
            d (list): Direction of the ray.
            a (list): One end of the rectangle side.
            b (list): Opposite end of the rectangle side.

        Returns:
            bool: True if the line intersects the rectangle side, else False.
        """
        v1 = [o[0] - a[0], o[1] - a[1]]
        v2 = [b[0] - a[0], b[1] - a[1]]
        v3 = [-d[1], d[0]]

        div = np.dot(v2, v3)

        if div == 0:
            return False

        t1 = np.linalg.norm(np.cross(v2, v1)) / div
        t2 = np.dot(v1, v3) / div

        if t1 >= 0 and 0 <= t2 <= 1:
            shot = Node((o[0] + t1 * d[0], o[1] + t1 * d[1]))
            dist_obs = self.get_dist(start, shot)
            dist_seg = self.get_dist(start, end)
            if dist_obs <= dist_seg:
                return True

        return False

    def is_intersect_circle(self, o, d, a, r):
        """
        Checks if a given line intersects a circular obstacle.

        Args:
            o (list): Origin of the ray.
            d (list): Direction of the ray.
            a (list): Center of the circle.
            r (float): Radius of the circle.

        Returns:
            bool: True if the line intersects the circular obstacle, else False.
        """
        d2 = np.dot(d, d)
        delta = self.delta

        if d2 == 0:
            return False

        t = np.dot([a[0] - o[0], a[1] - o[1]], d) / d2

        if 0 <= t <= 1:
            shot = Node((o[0] + t * d[0], o[1] + t * d[1]))
            if self.get_dist(shot, Node(a)) <= r + delta:
                return True

        return False

    def is_collision(self, start, end):
        """
        Checks if a given path intersects any obstacles.

        Args:
            start (Node): Start point of the path.
            end (Node): End point of the path.

        Returns:
            bool: True if the path intersects any obstacles, else False.
        """
        if self.is_inside_obs(start) or self.is_inside_obs(end):
            return True

        o, d = self.get_ray(start, end)
        obs_vertex = self.get_obs_vertex()

        # Check intersection with all rectangular obstacles' edges
        for (v1, v2, v3, v4) in obs_vertex:
            if self.is_intersect_rec(start, end, o, d, v1, v2) or \
               self.is_intersect_rec(start, end, o, d, v2, v3) or \
               self.is_intersect_rec(start, end, o, d, v3, v4) or \
               self.is_intersect_rec(start, end, o, d, v4, v1):
                return True

        # Check intersection with all circular obstacles
        for (x, y, r) in self.obs_circle:
            if self.is_intersect_circle(o, d, [x, y], r):
                return True

        return False

    def is_inside_obs(self, node):
        """
        Checks if a given node is inside any of the obstacles.

        Args:
            node (Node): Node to be checked.

        Returns:
            bool: True if the node is inside an obstacle, else False.
        """
        delta = self.delta

        # Check inside circular obstacles
        for (x, y, r) in self.obs_circle:
            if math.hypot(node.x - x, node.y - y) <= r + delta:
                return True

        # Check inside rectangular obstacles
        for (x, y, w, h) in self.obs_rectangle:
            if 0 <= node.x - (x - delta) <= w + 2 * delta \
                    and 0 <= node.y - (y - delta) <= h + 2 * delta:
                return True

        # Check inside boundary obstacles
        for (x, y, w, h) in self.obs_boundary:
            if 0 <= node.x - (x - delta) <= w + 2 * delta \
                    and 0 <= node.y - (y - delta) <= h + 2 * delta:
                return True

        return False

    @staticmethod
    def get_ray(start, end):
        """
        Computes the ray (origin and direction) between two nodes.

        Args:
            start (Node): Start node.
            end (Node): End node.

        Returns:
            orig (list): Origin of the ray.
            direc (list): Direction of the ray.
        """
        orig = [start.x, start.y]
        direc = [end.x - start.x, end.y - start.y]
        return orig, direc

    @staticmethod
    def get_dist(start, end):
        """
        Calculates the Euclidean distance between two nodes.

        Args:
            start (Node): Start node.
            end (Node): End node.

        Returns:
            float: Distance between the two nodes.
        """
        return math.hypot(end.x - start.x, end.y - start.y)


class Env:
    def __init__(self):
        """
        Initializes the environment with obstacle boundaries, circles, and rectangles.
        """
        self.x_range = (0, 50)
        self.y_range = (0, 30)
        self.obs_boundary = self.obs_boundary()
        self.obs_circle = self.obs_circle()
        self.obs_rectangle = self.obs_rectangle()

    @staticmethod
    def obs_boundary():
        """
        Returns the boundary obstacles.

        Returns:
            obs_boundary (list): List of boundary obstacles represented as [x, y, width, height].
        """
        obs_boundary = [
            [0, 0, 1, 30],
            [0, 30, 50, 1],
            [1, 0, 50, 1],
            [50, 1, 1, 30]
        ]
        return obs_boundary

    @staticmethod
    def obs_rectangle():
        """
        Returns the rectangular obstacles.

        Returns:
            obs_rectangle (list): List of rectangular obstacles represented as [x, y, width, height].
        """
        obs_rectangle = [
            [5, 18, 2, 7],
            [7, 23, 7, 2],
            [13, 3, 4, 4],
            [19, 3, 4, 4],
            [25, 3, 4, 4],
            [31, 3, 4, 4],
            [31, 9, 4, 4],
            [31, 15, 4, 4],
            [31, 21, 4, 4],
            [16, 10, 2, 15],
            [20, 10, 10, 2],
            [28, 12, 2, 5],
        ]
        return obs_rectangle

    @staticmethod
    def obs_circle():
        """
        Returns the circular obstacles.

        Returns:
            obs_cir (list): List of circular obstacles represented as [x, y, radius].
        """
        obs_cir = [
            [42, 17, 5],
            [25, 15, 2],
            [10, 20, 2]
        ]

        return obs_cir

class Node:
    def __init__(self, n):
        """
        Initializes a node object with coordinates and no parent.

        Args:
            n (tuple): Coordinates of the node (x, y).
        """
        self.x = n[0]
        self.y = n[1]
        self.parent = None


class Plotting:
    def __init__(self, x_start, x_goal):
        """
        Initializes the plotting class for visualizing the environment and paths.

        Args:
            x_start (tuple): Starting coordinates.
            x_goal (tuple): Goal coordinates.
        """
        self.xI, self.xG = x_start, x_goal
        self.env = Env()
        self.obs_bound = self.env.obs_boundary
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle

    def animation(self, nodelist, path, name, animation=False):
        """
        Animates the exploration and final path.

        Args:
            nodelist (list): List of visited nodes.
            path (list): Path found by the algorithm.
            name (str): Title of the plot.
            animation (bool): Whether to animate the exploration process.
        """
        self.plot_grid(name)
        self.plot_visited(nodelist, animation)
        self.plot_path(path)

    def animation_connect(self, V1, V2, path, name):
        """
        Animates the bidirectional exploration and final path.

        Args:
            V1 (list): First tree of visited nodes.
            V2 (list): Second tree of visited nodes.
            path (list): Path found by the algorithm.
            name (str): Title of the plot.
        """
        self.plot_grid(name)
        self.plot_visited_connect(V1, V2)
        self.plot_path(path)

    def plot_grid(self, name):
        """
        Plots the obstacles and start/goal points on a grid.

        Args:
            name (str): Title of the plot.
        """
        fig, ax = plt.subplots()

        # Plot boundary obstacles
        for (ox, oy, w, h) in self.obs_bound:
            ax.add_patch(
                patches.Rectangle(
                    (ox, oy), w, h,
                    edgecolor='black',
                    facecolor='black',
                    fill=True
                )
            )

        # Plot rectangular obstacles
        for (ox, oy, w, h) in self.obs_rectangle:
            ax.add_patch(
                patches.Rectangle(
                    (ox, oy), w, h,
                    edgecolor='black',
                    facecolor='black',
                    fill=True
                )
            )

        # Plot circular obstacles
        for (ox, oy, r) in self.obs_circle:
            ax.add_patch(
                patches.Circle(
                    (ox, oy), r,
                    edgecolor='black',
                    facecolor='black',
                    fill=True
                )
            )

        plt.plot(self.xI[0], self.xI[1], "bs", linewidth=3)
        plt.plot(self.xG[0], self.xG[1], "gs", linewidth=3)

        plt.title(name)
        plt.axis("equal")

    @staticmethod
    def plot_visited(nodelist, animation):
        """
        Plots visited nodes during exploration.

        Args:
            nodelist (list): List of visited nodes.
            animation (bool): Whether to animate the exploration process.
        """
        if animation:
            count = 0
            for node in nodelist:
                count += 1
                if node.parent:
                    plt.plot([node.parent.x, node.x], [node.parent.y, node.y], "-g")
                    plt.gcf().canvas.mpl_connect('key_release_event',
                                                 lambda event:
                                                 [exit(0) if event.key == 'escape' else None])
                    if count % 10 == 0:
                        plt.pause(0.001)
        else:
            for node in nodelist:
                if node.parent:
                    plt.plot([node.parent.x, node.x], [node.parent.y, node.y], "-g")

    @staticmethod
    def plot_visited_connect(V1, V2):
        """
        Plots visited nodes for bidirectional trees.

        Args:
            V1 (list): First tree of visited nodes.
            V2 (list): Second tree of visited nodes.
        """
        len1, len2 = len(V1), len(V2)

        for k in range(max(len1, len2)):
            if k < len1:
                if V1[k].parent:
                    plt.plot([V1[k].x, V1[k].parent.x], [V1[k].y, V1[k].parent.y], "-r")
            if k < len2:
                if V2[k].parent:
                    plt.plot([V2[k].x, V2[k].parent.x], [V2[k].y, V2[k].parent.y], "-b")

            plt.gcf().canvas.mpl_connect('key_release_event',
                                         lambda event: [exit(0) if event.key == 'escape' else None])

            if k % 2 == 0:
                plt.pause(0.001)

        plt.pause(0.01)

    @staticmethod
    def plot_path(path):
        """
        Plots the final path.

        Args:
            path (list): List of coordinates forming the path.
        """
        if len(path) != 0:
            plt.plot([x[0] for x in path], [x[1] for x in path], '-g', linewidth=2)
        plt.pause(0.001)
        plt.show(block=True)
        plt.close()


class RrtConnect:
    def __init__(self, s_start, s_goal, step_len, goal_sample_rate, iter_max):
        """
        Initializes the RRT-Connect algorithm parameters.

        Args:
            s_start (tuple): Starting coordinates.
            s_goal (tuple): Goal coordinates.
            step_len (float): Step length for expansion.
            goal_sample_rate (float): Probability of sampling directly towards the goal.
            iter_max (int): Maximum number of iterations.
        """
        self.s_start = Node(s_start)
        self.s_goal = Node(s_goal)
        self.step_len = step_len
        self.goal_sample_rate = goal_sample_rate
        self.iter_max = iter_max
        self.V1 = [self.s_start]
        self.V2 = [self.s_goal]

        self.env = Env()
        self.plotting = Plotting(s_start, s_goal)
        self.utils = Utils()

        self.x_range = self.env.x_range
        self.y_range = self.env.y_range
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary

    def planning(self):
        """
        Plans a path using the RRT-Connect algorithm.

        Returns:
            tuple: The planned path and its total length if a path is found, otherwise None.
        """
        for i in range(self.iter_max):
            node_rand = self.generate_random_node(self.s_goal, self.goal_sample_rate)
            node_near = self.nearest_neighbor(self.V1, node_rand)
            node_new = self.new_state(node_near, node_rand)

            if node_new and not self.utils.is_collision(node_near, node_new):
                self.V1.append(node_new)
                node_near_prim = self.nearest_neighbor(self.V2, node_new)
                node_new_prim = self.new_state(node_near_prim, node_new)

                if node_new_prim and not self.utils.is_collision(node_new_prim, node_near_prim):
                    self.V2.append(node_new_prim)

                    # Incrementally grow the connection between V1 and V2
                    while True:
                        node_new_prim2 = self.new_state(node_new_prim, node_new)
                        if node_new_prim2 and not self.utils.is_collision(node_new_prim2, node_new_prim):
                            self.V2.append(node_new_prim2)
                            node_new_prim = self.change_node(node_new_prim, node_new_prim2)
                        else:
                            break

                        if self.is_node_same(node_new_prim, node_new):
                            break

                if self.is_node_same(node_new_prim, node_new):
                    return self.extract_path(node_new, node_new_prim)

            if len(self.V2) < len(self.V1):
                list_mid = self.V2
                self.V2 = self.V1
                self.V1 = list_mid

        return None

    @staticmethod
    def change_node(node_new_prim, node_new_prim2):
        """
        Updates the parent relationship between two nodes.

        Args:
            node_new_prim (Node): Existing parent node.
            node_new_prim2 (Node): New child node.

        Returns:
            Node: New node with the updated parent relationship.
        """
        node_new = Node((node_new_prim2.x, node_new_prim2.y))
        node_new.parent = node_new_prim

        return node_new

    @staticmethod
    def is_node_same(node_new_prim, node_new):
        """
        Checks if two nodes are the same by comparing coordinates.

        Args:
            node_new_prim (Node): First node.
            node_new (Node): Second node.

        Returns:
            bool: True if the nodes are the same, else False.
        """
        if node_new_prim.x == node_new.x and \
                node_new_prim.y == node_new.y:
            return True

        return False

    def generate_random_node(self, sample_goal, goal_sample_rate):
        """
        Generates a random node within the environment's range.

        Args:
            sample_goal (Node): Goal node to sample towards.
            goal_sample_rate (float): Probability of sampling directly towards the goal.

        Returns:
            Node: Randomly generated node.
        """
        delta = self.utils.delta

        if np.random.random() > goal_sample_rate:
            return Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                         np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))

        return sample_goal

    @staticmethod
    def nearest_neighbor(node_list, n):
        """
        Finds the nearest node in the given list to a target node.

        Args:
            node_list (list): List of nodes to search in.
            n (Node): Target node to compare distances to.

        Returns:
            Node: Nearest node to the target node.
        """
        return node_list[int(np.argmin([math.hypot(nd.x - n.x, nd.y - n.y)
                                        for nd in node_list]))]

    def new_state(self, node_start, node_end):
        """
        Computes a new node by moving from a start node towards an end node.

        Args:
            node_start (Node): Starting node.
            node_end (Node): Target node.

        Returns:
            Node: New node positioned towards the target node.
        """
        dist, theta = self.get_distance_and_angle(node_start, node_end)

        dist = min(self.step_len, dist)
        node_new = Node((node_start.x + dist * math.cos(theta),
                         node_start.y + dist * math.sin(theta)))
        node_new.parent = node_start

        return node_new

    @staticmethod
    def extract_path(node_new, node_new_prim):
        """
        Extracts the path connecting two trees.

        Args:
            node_new (Node): Last node of the first tree.
            node_new_prim (Node): Last node of the second tree.

        Returns:
            tuple: Path connecting the two trees and its total length.
        """
        path1 = [(node_new.x, node_new.y)]
        node_now = node_new

        while node_now.parent is not None:
            node_now = node_now.parent
            path1.append((node_now.x, node_now.y))

        path2 = [(node_new_prim.x, node_new_prim.y)]
        node_now = node_new_prim

        while node_now.parent is not None:
            node_now = node_now.parent
            path2.append((node_now.x, node_now.y))

        combined_path = list(list(reversed(path1)) + path2)
        total_length = sum(math.hypot(combined_path[i][0] - combined_path[i + 1][0], combined_path[i][1] - combined_path[i + 1][1]) for i in range(len(combined_path) - 1))
        return combined_path, total_length

    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        """
        Computes the distance and angle between two nodes.

        Args:
            node_start (Node): Starting node.
            node_end (Node): Target node.

        Returns:
            tuple: Distance and angle between the two nodes.
        """
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)


def main():
    """
    Main function to run the RRT-Connect algorithm and plot the results.
    """
    x_start = (2, 2)  # Starting node
    x_goal = (48, 24)  # Goal node

    rrt_conn = RrtConnect(x_start, x_goal, 0.8, 0.05, 5000)

    start_time = time.time()
    path, length = rrt_conn.planning()

    rrt_conn.plotting.animation_connect(rrt_conn.V1, rrt_conn.V2, path, "RRT_CONNECT")

    end_time = time.time()

    print(f"{end_time - start_time} \t {length}")

if __name__ == '__main__':
    main()
