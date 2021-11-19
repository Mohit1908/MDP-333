SZE = [5,5]
import numpy as np
import matplotlib.pyplot as plt


EXIT_LOCATION = np.array([-1,-1])
N = np.array([1,0])
E = np.array([0,1])
W = np.array([0,-1])
S = np.array([-1,0])

depots = {'R':np.array([4,0]),'Y':np.array([0,0]),'B':np.array([0,3]),'G':np.array([4,4])}

walls = {}
walls['N'] = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[1,1,1,1,1]])
walls['S'] = np.array([[1,1,1,1,1],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]])
walls['E'] = np.array([[1,0,1,0,1],[1,0,1,0,1],[0,0,0,0,1],[0,1,0,0,1],[0,1,0,0,1]])
walls['W'] = np.array([[1,1,0,1,0],[1,1,0,1,0],[1,0,0,0,0],[1,0,1,0,0],[1,0,1,0,0]])

class state:
	def __init__(self,location,has_passenger = 0):
		self.location = location
		self.has_passenger = has_passenger

pickup = 'R'
drop = 'Y'

start = np.array([3,0])
has_passenger = 0

actions = ('N','E','W','S','PickUp','PutDown')

def parta1():
	def transition_model(state1,action,state2):
		p = 0.0
		if(action == 'PickUp'):
			if(state2.location == state1.location).all():
				p = 1.0
		elif(action == 'PutDown'):
			if(state2.location == EXIT_LOCATION):
				p = 1.0
		else:
			if(state1.location[0] != state2.location[0]):
				if(state1.location[1]==state2.location[1]):
					if(state2.location[0] - state1.location[0] == 1):
						if(walls['N'][state1.location[0]][state1.location[1]] == 0):
							if(action == 'N'):
								p = 0.85
							else:
								p = 0.05
					if(state1.location[0] - state2.location[0] == 1):
						if(walls['S'][state1.location[0]][state1.location[1]] == 0):
							if(action == 'S'):
								p = 0.85
							else:
								p = 0.05
			elif(state1.location[1] != state2.location[1]):
				if(state1.location[0]==state2.location[0]):
					if(state2.location[1] - state1.location[1] == 1):
						if(walls['E'][state1.location[0]][state1.location[1]] == 0):
							if(action == 'E'):
								p = 0.85
							else:
								p = 0.05
					if(state1.location[1] - state2.location[1] == 1):
						if(walls['W'][state1.location[0]][state1.location[1]] == 0):
							if(action == 'W'):
								p = 0.85
							else:
								p = 0.05
			else:
				if(walls['W'][state1.location[0]][state1.location[1]] == 1):
					if(action == 'W'):
						p += 0.85
					else:
						p+=0.05
				if(walls['N'][state1.location[0]][state1.location[1]] == 1):
					if(action == 'N'):
						p += 0.85
					else:
						p+=0.05
				if(walls['E'][state1.location[0]][state1.location[1]] == 1):
					if(action == 'E'):
						p += 0.85
					else:
						p+=0.05
				if(walls['S'][state1.location[0]][state1.location[1]] == 1):
					if(action == 'S'):
						p += 0.85
					else:
						p+=0.05
				
		return p

	def reward_model(state1,action,state2):
		reward = -1
		if(action == 'PutDown'):
			if(state1.location == depots[drop]).all():
				reward = 20
			else:
				reward = -10
		if(action == 'PickUp'):
			if(state1.location != depots[pickup]).any():
				reward = -10
		return reward