SZE = [5,5]
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.numeric import Inf


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

class parta1:
	def __init__(self,start,pickup,drop):
		self.start = start
		self.pickup = pickup
		self.drop = drop

	def transition_model(self,state1,action,state2):
		p = 0.0
		if(action == 'PickUp'):
			if(state2.location == state1.location).all():
				p = 1.0
		elif(action == 'PutDown'):
			if(state2.location == EXIT_LOCATION):
				p = 1.0
		elif action in (['N','W','E','S']):
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

	def next_state(self,state1,action):
		if(action == 'PickUp'):
			if(state1.location == self.pickup):
				state1.has_passenger = 1
				return {state1:1}
			else:
				return {state1:1}
		if(action == 'Drop'):
			if(state1.location == self.drop):
				if(state1.has_passenger == 1):
					state1.location = EXIT_LOCATION
					return {state1:1}
				else:
					return {state1:1}
			else:
				return {state1:1}
		s1 = state([state1.location[0],state1.location[1]],state1.has_passenger)
		s2 = state([state1.location[0],state1.location[1]+1],state1.has_passenger)
		s3 = state([state1.location[0],state1.location[1]-1],state1.has_passenger)
		s4 = state([state1.location[0]+1,state1.location[1]],state1.has_passenger)
		s5 = state([state1.location[0]-1,state1.location[1]],state1.has_passenger)

		return {s1:self.transition_model(state1,action,s1),s2:self.transition_model(state1,action,s2),s3:self.transition_model(state1,action,s3),s4:self.transition_model(state1,action,s4),s5:self.transition_model(state1,action,s5),}
			
	def reward_model(self,state1,action,state2):
		reward = -1
		if(action == 'PutDown'):
			if(state1.location == depots[self.drop]).all():
				reward = 20
			else:
				reward = -10
		if(action == 'PickUp'):
			if(state1.location != depots[pickup]).any():
				reward = -10
		return reward

def value_iter(p,eps,discount_factor):
	value1 = np.zeros((SZE[0],SZE[1],2))
	value2 = np.zeros((SZE[0],SZE[1],2))
	policy = np.empty((SZE[0],SZE[1],2),dtype = str)
	achieved_eps = Inf
	while(achieved_eps > eps):
		achieved_eps = 0
		for i in range(SZE[0]):
			for j in range(SZE[1]):
				for passenger in range(2):
					st = state(np.array([i,j]),passenger)
					maxx = -Inf
					for a in actions:
						val = 0
						d = p.next_state(st,a)
						for s2 in d:
							val += d[s2]*(p.reward_model(st,a,s2) + discount_factor*value1[s2.location[0],s2.location[1],s2.has_passenger])
						maxx = max(val,maxx)
					value2[st.location[0],st.location[1],st.has_passenger] = maxx
					achieved_eps = max(achieved_eps,abs(maxx-value1[st.location[0],st.location[1],st.has_passenger]))
		value1 = value2
	def extract_policy(value):
		pass
	return extract_policy(value2)
	
def policy_iter(p,discount_factor):
	policy1 = np.empty((SZE[0],SZE[1],2),dtype = str)
	policy2 = np.empty((SZE[0],SZE[1],2),dtype = str)
	changed = True
	def extract_value(policy):
		pass
	def extract_policy(value):
		pass
	while(changed):
		value = extract_value(policy1)
		policy2 = extract_policy(value)
		if((policy1 != policy2).any()):
			changed = True
		policy1 = policy2
	return policy1


	

p = parta1(start,pickup,drop)
s1 = state(start)
s2 = state(np.array([4,0]))
print(p.transition_model(s1,'N',s2))
print(p.reward_model(s1,'N',s2))
print(p.next_state(s1,'N'))
d = p.next_state(s1,'N')

for k in d:
	print(k.location)

value_iter(2,1)