import numpy as np

HEIGHT = 10
WIDTH = 10
TIMESTEPS = (HEIGHT+WIDTH)*4


class Maze(object):
	def __init__(self, height, width):
		self.height = height
		self.width = width
	 	self.reset()

	def generate_path(self, momentum=False):
		start_set = set([])
		end_set = set([])

		def random_valid_neighbor(coord, momentum=False, dir=None):
			x, y = coord

			N, S = (x, y - 1), (x, y + 1)
			E, W = (x - 1, y), (x + 1, y)

			neighbors = [N, S, E, W]
			valid = [0]*4

			if y - 1 >= 0:
				valid[0] = 1
			if y + 1 <= self.height - 1:
				valid[1] = 1
			if x - 1 >= 0:
				valid[2] = 1
			if x + 1 <= self.width - 1:
				valid[3] = 1

			if momentum and dir is not None and valid[dir] != 0:
				probs = np.array(valid, dtype=np.float) * 1.0/(np.sum(valid) + 1)
				probs[dir] = 2 * 1.0/(np.sum(valid) + 1)
			else:
				probs = probs = np.array(valid, dtype=np.float) * 1.0/np.sum(valid)

			i = np.random.choice(len(neighbors), 1, p=probs)[0]
			return neighbors[i]

		def get_direction(prev, curr):
			xp, yp = prev
			xc, yc = curr
			if xc - xp == 1:
				return 0
			if xc - xp == -1:
				return 1
			if yc - yp == 1:
				return 2
			if yc - yp == -1:
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
		self.maze = np.ones(self.height*self.width).reshape((self.height, self.width))

	def display(self):
		print(self.maze)
		print("\n")


if __name__ == "__main__":

	maze = Maze(HEIGHT, WIDTH)
	maze.generate_path()
	maze.display()

	maze.reset()

	maze.generate_path(True)
	maze.display()

	# for i in timesteps:
