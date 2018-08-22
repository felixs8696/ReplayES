import numpy as np
np.warnings.filterwarnings('ignore')
from copy import deepcopy

from RandomActor import RandomActor

HEIGHT = 10
WIDTH = 10
TIMESTEPS = (HEIGHT+WIDTH)*4


class Maze(object):
	def __init__(self, height, width):
		self.height = height
		self.width = width
		self.start = (0,0)
		self.goal = (self.height - 1, self.width - 1)
	 	self.reset()

	def generate_path(self, momentum=False):
		start_set = set([])
		end_set = set([])

		def random_valid_neighbor(coord, momentum=False, dir=None):
			h, w = coord

			N, S = (h - 1, w), (h + 1, w)
			E, W = (h, w - 1), (h, w + 1)

			neighbors = [N, S, E, W]
			valid = [0]*4

			if h - 1 >= 0:
				valid[0] = 1
			if h + 1 <= self.height - 1:
				valid[1] = 1
			if w - 1 >= 0:
				valid[2] = 1
			if w + 1 <= self.width - 1:
				valid[3] = 1

			if momentum and dir is not None and valid[dir] != 0:
				probs = np.array(valid, dtype=np.float) * 1.0/(np.sum(valid) + 1)
				probs[dir] = 2 * 1.0/(np.sum(valid) + 1)
			else:
				probs = probs = np.array(valid, dtype=np.float) * 1.0/np.sum(valid)

			i = np.random.choice(len(neighbors), 1, p=probs)[0]
			return neighbors[i]

		def get_direction(prev, curr):
			hp, wp = prev
			hc, wc = curr
			if hc - hp == -1: # N
				return 0
			if hc - hp == 1: # S
				return 1
			if wc - wp == -1: # E
				return 2
			if wc - wp == 1: # W
				return 3

		start_head = (0,0)
		end_head = (self.height - 1, self.width - 1)
		last_start_dir = None
		last_end_dir = None
		start_set.add(start_head)
		self.maze[start_head] = 0
		end_set.add(end_head)
		self.maze[end_head] = 0

		solved = False

		while not solved:
			start_prop = random_valid_neighbor(start_head, momentum=momentum, dir=last_start_dir)
			if start_prop in end_set:
				start_set.add(start_prop)
				self.maze[start_prop] = 0
				solved = True
				break
			else:
				start_set.add(start_prop)
				self.maze[start_prop] = 0
			last_start_dir = get_direction(start_head, start_prop)
			start_head = start_prop

			end_prop = random_valid_neighbor(end_head, momentum=momentum, dir=last_end_dir)
			if end_prop in start_set:
				end_set.add(end_prop)
				self.maze[end_prop] = 0
				solved = True
				break
			else:
				end_set.add(end_prop)
				self.maze[end_prop] = 0
			last_end_dir = get_direction(end_head, end_prop)
			end_head = end_prop

	def reset(self):
		self.maze = np.ones(self.height*self.width).reshape((self.height, self.width)) * -1

	def display(self):
		print(self.maze)
		print("\n")

	def get_flat_index(self, pos):
		h, w = pos
		return h * self.width + w

	def get_maze_index(self, flat_index):
		h = flat_index / self.width
		w = flat_index % self.width
		return (h, w)

	def display_stddevs(self, sd_map):
		std_devs = np.zeros(self.height*self.width).reshape((self.height, self.width))
		for i in range(self.height * self.width):
			# print(self.get_maze_index(i), ":", sd_map.map[i])
			if self.maze[self.get_maze_index(i)] == -1:
				std_devs[self.get_maze_index(i)] = -1
			# elif len(sd_map.map[i]) == 0:
			# 	std_devs[self.get_maze_index(i)] = 0
			else:
				std_devs[self.get_maze_index(i)] = np.std(sd_map.map[i])
		print(np.round(std_devs, 2))
		return std_devs

class StdDevMap(object):
	def __init__(self, height, width):
		self.map = [[] for i in range(height*width)]

	def put(self, idx, value):
		if idx > len(self.map):
			print("Index: ", idx)
		self.map[idx].append(value)
		# print(self.map[idx])

if __name__ == "__main__":

	maze = Maze(HEIGHT, WIDTH)
	# maze.generate_path()
	# maze.display()

	# maze.reset()

	maze.generate_path(True)
	maze.display()

	sd_map = StdDevMap(maze.height, maze.width)
	actors = [RandomActor(maze.goal, i) for i in range(10)]


	for i in range(len(actors)):
		actor = actors[i]
		idx_traversed = []
		for t in range(TIMESTEPS):
			idx_traversed.append(maze.get_flat_index(actor.pos))
			new_pos = actor.act(maze)
			new_reward = actor.update_reward()
		# print(actor.reward)
		# print(idx_traversed)
		for idx in idx_traversed:
			sd_map.put(idx, actor.reward)

	maze.display_stddevs(sd_map)

