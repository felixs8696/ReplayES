import numpy as np

INPUTS = 4
OUTPUTS = 4

def euclidean_distance(a, b):
	return np.linalg.norm(np.array(a) - np.array(b))

class RandomActor(object):
	def __init__(self, goal, id=0):
		self.goal = goal
		self.id = id
		self.pos = (0,0)
		self.reward = 0

	def act(self, maze):
		h, w = self.pos

		valid = [self.pos]

		if h - 1 >= 0 and maze.maze[(h - 1, w)] != -1:
			valid.append((h - 1, w))
		if h + 1 <= maze.height - 1 and maze.maze[(h + 1, w)] != -1:
			valid.append((h + 1, w))
		if w - 1 >= 0 and maze.maze[(h, w - 1)] != -1:
			valid.append((h, w - 1))
		if w + 1 <= maze.width - 1 and maze.maze[(h, w + 1)] != -1:
			valid.append((h, w + 1))

		i = np.random.choice(len(valid), 1)[0]
		self.pos = valid[i]

		return (h, w)

	def update_reward(self):
		# TODO: Discount future reward not past
		prev_reward = self.reward
		curr_reward = euclidean_distance(self.pos, self.goal)
		discounted_reward = .9 * self.reward
		self.reward = curr_reward + discounted_reward
		return self.reward - prev_reward