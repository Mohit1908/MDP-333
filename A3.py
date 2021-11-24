import numpy as np
import matplotlib.pyplot as plt
from numpy.core.numeric import Inf
import random
import matplotlib.pyplot as plt

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
		coin = random.random()
		p = 0
		for s in d:
			if(p + d[s] >= coin):
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


def display(state,p):
	for i in range(SZE[0]):
		for j in range(SZE[1]):
			ni = SZE[0]-i-1
			if(state.location[0] == ni and state.location[1] == j and state.has_passenger == 1):
				print("T",end = "")   #Taxi has passenger
			elif(state.location[0] == ni and state.location[1] == j and state.has_passenger == 0):
				print("t",end = "")	  #Empty Taxi...
			elif((ni == depots[p.pickup][0]) and (j == depots[p.pickup][1])):
				print("S",end = "")
			elif(ni == depots[p.drop][0] and j == depots[p.drop][1]):
				print("D",end = "")
			else:
				print("0",end = "")
		print("")
	print("")

def value_iter(p,eps,discount_factor):

	value1 = np.zeros((SZE[0],SZE[1],2))
	value2 = np.zeros((SZE[0],SZE[1],2))

	achieved_eps = Inf

	iterations = 0
	while(achieved_eps > eps):
		achieved_eps = 0

		for i in range(SZE[0]):
			for j in range(SZE[1]):
				for passenger in range(2):

					st = state(np.array([i,j]),passenger)
					maxx = -Inf

					for a in actions:
						#print(a)
						val = 0
						d = p.next_state(st,a)

						for s2 in (d.keys()):
							val += d[s2]*(p.reward_model(st,a,s2) + discount_factor*valueFunction(value1,s2))
						maxx = max(val,maxx)
					
					value2[st.location[0],st.location[1],st.has_passenger] = maxx
					achieved_eps = max(achieved_eps,abs(maxx-value1[st.location[0],st.location[1],st.has_passenger]))
		
		value1 = value2.copy()
		iterations += 1
		#print_value(value1)
		#print(achieved_eps)

	print("The number of iterations taken are : "+str(iterations))
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

	iterations = 0
	while(changed):
		value = extract_value(policy1,p,eps,discount_factor)
		policy2 = extract_policy(value,p,discount_factor)
		if((policy1 != policy2).any()):
			changed = True
		else:
			changed = False
		policy1 = policy2.copy()
		iterations += 1
	
	print("The number of iterations taken are : "+str(iterations))
	return policy1

def q_learning(p,alpha,discount_factor,epsilon,exponential_decay = False):

	MAX_EPISODES = 2000
	MAX_ITERATIONS = 500

	episode = 1
	Q = np.zeros((SZE[0],SZE[1],2,6)) # 6 actions
	initial_start = p.start

	reward_iterations = []
	while(episode<MAX_EPISODES):

		iterations = 1
		curr_state = state(np.random.randint(5,size = (2,)))
		p.start = curr_state

		discounted_rewards = 0
		thisFactor = 1
		while(iterations<MAX_ITERATIONS):

			this_eps = epsilon if not exponential_decay else (epsilon)/iterations

			if(random.random()<this_eps):
				this_action_num = random.randint(0,5)
			else:
				this_action_num = Q[curr_state.location[0]][curr_state.location[1]][curr_state.has_passenger].argmax()

			#print(this_action_num)
			next_state = p.ret_next_state(curr_state,actions[this_action_num])
			this_reward = p.reward_model(curr_state,actions[this_action_num],next_state)

			discounted_rewards += this_reward*thisFactor
			thisFactor *= discount_factor

			if(next_state.location[0]==-1 and next_state.location[1]==-1):
				Q[curr_state.location[0]][curr_state.location[1]][curr_state.has_passenger][this_action_num] += alpha*(this_reward - Q[curr_state.location[0]][curr_state.location[1]][curr_state.has_passenger][this_action_num])
				break
			else:
				Q[curr_state.location[0]][curr_state.location[1]][curr_state.has_passenger][this_action_num] += alpha*(this_reward + discount_factor*np.max(Q[next_state.location[0]][next_state.location[1]][next_state.has_passenger]) - Q[curr_state.location[0]][curr_state.location[1]][curr_state.has_passenger][this_action_num])

			curr_state = state([next_state.location[0],next_state.location[1]],next_state.has_passenger)
			iterations += 1

		reward_iterations.append(discounted_rewards)
		#print(iterations)
		episode+=1

	#plt.plot(reward_iterations)
	#plt.show()
	policy = np.empty((SZE[0],SZE[1],2),dtype = 'object')
	for i in range(SZE[0]):
			for j in range(SZE[1]):
				for passenger in range(2):

					policy[i][j][passenger] = actions[Q[i][j][passenger].argmax()]

	p.start = initial_start
	return policy

def sarsa_learning(p,alpha,discount_factor,epsilon,exponential_decay = False):

	MAX_EPISODES = 2000
	MAX_ITERATIONS = 500

	episode = 1
	Q = np.zeros((SZE[0],SZE[1],2,6)) # 6 actions
	initial_start = p.start

	reward_iterations = []
	while(episode<MAX_EPISODES):

		iterations = 1
		curr_state = state(np.random.randint(5,size = (2,)))
		p.start = curr_state

		discounted_rewards = 0
		thisFactor = 1

		this_action_num = 100
		next_action_num = 100

		while(iterations<MAX_ITERATIONS):

			this_eps = epsilon if not exponential_decay else (epsilon)/iterations

			if (this_action_num!=100):
				this_action_num = next_action_num
			elif(random.random()<this_eps):
				this_action_num = random.randint(0,5)
			else:
				this_action_num = Q[curr_state.location[0]][curr_state.location[1]][curr_state.has_passenger].argmax()

			next_state = p.ret_next_state(curr_state,actions[this_action_num])
			this_reward = p.reward_model(curr_state,actions[this_action_num],next_state)

			discounted_rewards += this_reward*thisFactor
			thisFactor *= discount_factor

			if(random.random()<this_eps):
				next_action_num = random.randint(0,5)
			else:
				next_action_num = Q[next_state.location[0]][next_state.location[1]][next_state.has_passenger].argmax()

			if(next_state.location[0]==-1 and next_state.location[1]==-1):
				Q[curr_state.location[0]][curr_state.location[1]][curr_state.has_passenger][this_action_num] += alpha*(this_reward - Q[curr_state.location[0]][curr_state.location[1]][curr_state.has_passenger][this_action_num])
				break
			else:
				Q[curr_state.location[0]][curr_state.location[1]][curr_state.has_passenger][this_action_num] += alpha*(this_reward + discount_factor*(Q[next_state.location[0]][next_state.location[1]][next_state.has_passenger][next_action_num]) - Q[curr_state.location[0]][curr_state.location[1]][curr_state.has_passenger][this_action_num])

			curr_state = state([next_state.location[0],next_state.location[1]],next_state.has_passenger)
			iterations += 1

		#print(iterations)
		reward_iterations.append(discounted_rewards)
		episode+=1

	#plt.plot(reward_iterations)
	#plt.show()
	policy = np.empty((SZE[0],SZE[1],2),dtype = 'object')
	for i in range(SZE[0]):
			for j in range(SZE[1]):
				for passenger in range(2):

					policy[i][j][passenger] = actions[Q[i][j][passenger].argmax()]

	p.start = initial_start
	return policy

def returnDisRewards(policy,p,discount_factor):			# Can also be used for running....simulation...
	TOTAL_RUNS = 10
	TOTAL_ITERATIONS = 500
	curr_run = 0
	totalRewards = 0

	initial_start = p.start

	while(curr_run<TOTAL_RUNS):
		print('New simulation episode--------------------')
		curr_state = state(np.random.randint(5,size = (2,)))
		p.start = curr_state
		iterations = 0
		thisFactor = 1

		while(iterations<TOTAL_ITERATIONS):
			display(curr_state,p)
			thisAction = policy[curr_state.location[0]][curr_state.location[1]][curr_state.has_passenger]
			next_state = p.ret_next_state(curr_state,thisAction)
			this_reward = p.reward_model(curr_state,thisAction,next_state)

			totalRewards += thisFactor*this_reward
			curr_state = state([next_state.location[0],next_state.location[1]],next_state.has_passenger)

			if(curr_state.location[0]==-1 and curr_state.location[1]==-1):
				break
			thisFactor *= discount_factor
			iterations += 1

		curr_run += 1

	p.start = initial_start
	return totalRewards/TOTAL_RUNS

if __name__ == "__main__":

	pickup = 'R'
	drop = 'Y'

	start = np.array([4,4])
	s1 = state(start)
	p = parta1(start,pickup,drop)
	#display(s1,p)
	
	this_policy = policy_iter(p,0.001,0.99)
	print_policy(this_policy)
	print(returnDisRewards(this_policy,p,0.99))