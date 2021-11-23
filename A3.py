import numpy as np
import matplotlib.pyplot as plt
from numpy.core.numeric import Inf
import random

SZE = [5,5]
EXIT_LOCATION = np.array([-1,-1])

N = np.array([1,0])
E = np.array([0,1])
W = np.array([0,-1])
S = np.array([-1,0])

actions = np.array(['N','E','W','S','PickUp','PutDown'])

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
			if(state2.location == state1.location).all():	#Here we assumed that PutDown at other than exit location doesn't make sense.
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
		s1 = state([state1.location[0],state1.location[1]],state1.has_passenger)
		if(action == 'PickUp'):
			if(state1.location == depots[self.pickup]).all():
				s1.has_passenger = 1
			return {s1:1}

		elif(action == 'PutDown'):
			if((state1.location == depots[self.drop]).all() and state1.has_passenger == 1):
				s1.location = EXIT_LOCATION
			return {s1:1}
		
		s1 = state([state1.location[0],state1.location[1]],state1.has_passenger)
		s2 = state([state1.location[0],state1.location[1]+1],state1.has_passenger)
		s3 = state([state1.location[0],state1.location[1]-1],state1.has_passenger)
		s4 = state([state1.location[0]+1,state1.location[1]],state1.has_passenger)
		s5 = state([state1.location[0]-1,state1.location[1]],state1.has_passenger)

		prob_dict =  {s1:self.transition_model(state1,action,s1),s2:self.transition_model(state1,action,s2),s3:self.transition_model(state1,action,s3),s4:self.transition_model(state1,action,s4),s5:self.transition_model(state1,action,s5),}

		return {x:y for x,y in prob_dict.items() if y!=0}
			
	def ret_next_state(self,state1,action):
		d = self.next_state(state1,action)
		n = random.randint(0,100)
		p = 0
		for s in d:
			if(p + d[s] >= n):
				return s
			else:
				p+= d[s]


	def reward_model(self,state1,action,state2):
		reward = -1
		if(action == 'PutDown'):
			if((state1.location == depots[self.drop]).all() and state1.has_passenger==1):
				reward = 20
			elif(state1.has_passenger==1):
				reward = -2						# Although this has prob=0
			else:
				reward = -10
		if(action == 'PickUp'):
			if((state1.location != depots[pickup]).any() and state1.has_passenger==0):
				reward = -10
		return reward

def valueFunction(value,state2):
	if(state2.location[0]==-1 and state2.location[1]==-1):
		return 0
	return value[state2.location[0],state2.location[1],state2.has_passenger]

def print_value(value):
	p = 0
	print("P = 0")
	for i in range(SZE[0]):
		for j in range(SZE[1]):
			print(value[SZE[0]-i-1][j][p],end = " ")
		print("")
	p = 1
	print("P = 1")
	for i in range(SZE[0]):
		for j in range(SZE[1]):
			print(value[SZE[0]-i-1][j][p],end = " ")
		print("")

def print_policy(policy):
	p = 0
	print("P = 0")
	for i in range(SZE[0]):
		for j in range(SZE[1]):
			print(policy[SZE[0]-i-1][j][p],end = " ")
		print("")
	p = 1
	print("P = 1")
	for i in range(SZE[0]):
		for j in range(SZE[1]):
			print(policy[SZE[0]-i-1][j][p],end = " ")
		print("")


def value_iter(p,eps,discount_factor):

	value1 = np.zeros((SZE[0],SZE[1],2))
	value2 = np.zeros((SZE[0],SZE[1],2))

	achieved_eps = Inf

	while(achieved_eps > eps):
		achieved_eps = 0

		for i in range(SZE[0]):
			for j in range(SZE[1]):
				for passenger in range(2):

					st = state(np.array([i,j]),passenger)
					maxx = -Inf

					for a in actions:
						print(a)
						val = 0
						d = p.next_state(st,a)

						for s2 in (d.keys()):
							val += d[s2]*(p.reward_model(st,a,s2) + discount_factor*valueFunction(value1,s2))
						maxx = max(val,maxx)
					
					value2[st.location[0],st.location[1],st.has_passenger] = maxx
					achieved_eps = max(achieved_eps,abs(maxx-value1[st.location[0],st.location[1],st.has_passenger]))
		
		value1 = value2.copy()
		print_value(value1)
		#print(achieved_eps)

	return extract_policy(value2,p,discount_factor)

def extract_policy(value,p,discount_factor):

	policy = np.empty((SZE[0],SZE[1],2),dtype = 'object')
	for i in range(SZE[0]):
		for j in range(SZE[1]):
			for passenger in range(2):
				st = state(np.array([i,j]),passenger)
				maxx = -Inf

				for a in actions:
					val = 0
					d = p.next_state(st,a)
					for s2 in (d.keys()):
						val += d[s2]*(p.reward_model(st,a,s2) + discount_factor*valueFunction(value,s2))
					if(maxx<val):
						policy[i][j][passenger] = a
						#print(a)
						#print(policy[i][j][passenger])
						maxx = val
	return policy

def extract_value(policy,p,eps,discount_factor):
	value1 = np.zeros((SZE[0],SZE[1],2))
	value2 = np.zeros((SZE[0],SZE[1],2))

	achieved_eps = Inf

	while(achieved_eps > eps):
		achieved_eps = 0

		for i in range(SZE[0]):
			for j in range(SZE[1]):
				for passenger in range(2):

					st = state(np.array([i,j]),passenger)
					val = 0
					d = p.next_state(st,policy[i][j][passenger])

					for s2 in (d.keys()):
						val += d[s2]*(p.reward_model(st,policy[i][j][passenger],s2) + discount_factor*valueFunction(value1,s2))
										
					value2[st.location[0],st.location[1],st.has_passenger] = val
					achieved_eps = max(achieved_eps,abs(val-value1[st.location[0],st.location[1],st.has_passenger]))
		
		value1 = value2.copy()
	return value1

def extract_value_linear_algebra(policy,p,eps,discount_factor):
	pass
	
def policy_iter(p,eps,discount_factor):

	policy1 = np.empty((SZE[0],SZE[1],2),dtype = 'object')
	policy2 = np.empty((SZE[0],SZE[1],2),dtype = 'object')
	changed = True

	#Initialize policy...
	for i in range(SZE[0]):
			for j in range(SZE[1]):
				for passenger in range(2):
					policy1[i][j][passenger] = random.choice(('N','E','W','S'))

	while(changed):
		value = extract_value(policy1,p,eps,discount_factor)
		policy2 = extract_policy(value,p,discount_factor)
		if((policy1 != policy2).any()):
			changed = True
		else:
			changed = False
		policy1 = policy2.copy()
	return policy1

if __name__ == "__main__":

	pickup = 'R'
	drop = 'Y'

	start = np.array([3,4])
		

	p = parta1(start,pickup,drop)
	s1 = state(start)
	s2 = state(np.array([2,4]))
	
	print(p.transition_model(s1,'N',s2))
	print(p.reward_model(s1,'N',s2))
	
	d = p.next_state(s1,'N')

	for k in d:
		print(k.location)

	this_policy = policy_iter(p,0.2,0.9)
	print_policy(this_policy)