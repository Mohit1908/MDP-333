import numpy as np
import matplotlib.pyplot as plt
from numpy.core.numeric import Inf
import random
import matplotlib.pyplot as plt
from os import system
import time

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
	def __init__(self,location,pickup,has_passenger = 0):
		self.location = location
		self.pickup = pickup
		self.has_passenger = has_passenger

class parta1:
	def __init__(self,start,drop):
		self.start = start
		#self.pickup = pickup
		self.drop = drop

	def transition_model(self,state1,action,state2):
		p = 0.0
		if(action == 'PickUp'):
			if(state2.location == state1.location).all():
				p = 1.0
		elif(action == 'PutDown'):		
			if(state2.location == EXIT_LOCATION).all():
				if(state1.location == depots[self.drop]).all():
					p = 1.0	
			elif(state2.location == state1.location).all():
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
		if(state1.has_passenger == 1):
			if(state1.location[0] != state1.pickup[0] or state1.location[1] != state1.pickup[1]):
				return {}
		s1 = state([state1.location[0],state1.location[1]],[state1.pickup[0],state1.pickup[1]],state1.has_passenger)
		if(action == 'PickUp'):
			if(state1.location[0] == state1.pickup[0] and state1.location[1] == state1.pickup[1]):
				s1.has_passenger = 1
			return {s1:1}

		elif(action == 'PutDown'):
			if((state1.location == depots[self.drop]).all()):
				if(state1.has_passenger == 1):
					s1.location = [EXIT_LOCATION[0],EXIT_LOCATION[1]]
			else:
				if(state1.has_passenger == 1):
					s1.pickup = [state1.location[0],state1.location[1]]
					s1.has_passenger = 0
			return {s1:1}
		
		s1 = state([state1.location[0],state1.location[1]],[state1.pickup[0],state1.pickup[1]],state1.has_passenger)
		s2 = state([state1.location[0],state1.location[1]+1],[state1.pickup[0],state1.pickup[1]],state1.has_passenger)
		s3 = state([state1.location[0],state1.location[1]-1],[state1.pickup[0],state1.pickup[1]],state1.has_passenger)
		s4 = state([state1.location[0]+1,state1.location[1]],[state1.pickup[0],state1.pickup[1]],state1.has_passenger)
		s5 = state([state1.location[0]-1,state1.location[1]],[state1.pickup[0],state1.pickup[1]],state1.has_passenger)
		if(state1.has_passenger == 1):
			s2.pickup = [s2.location[0],s2.location[1]]
			s3.pickup = [s3.location[0],s3.location[1]]
			s4.pickup = [s4.location[0],s4.location[1]]
			s5.pickup = [s5.location[0],s5.location[1]]

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
			if(state1.has_passenger == 0):
				reward = -10
		if(action == 'PickUp'):
			if(((state1.location[0] != state1.pickup[0]) or (state1.location[1] != state1.pickup[1])) and state1.has_passenger==0):
				reward = -10
		return reward

def valueFunction(value,state2):
	if(state2.location[0]==-1 and state2.location[1]==-1):
		return 0
	return value[state2.location[0]][state2.location[1]][state2.pickup[0]][state2.pickup[1]][state2.has_passenger]

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
	#time.sleep(1)
	#system("clear")
	for i in range(SZE[0]):
		for j in range(SZE[1]):
			ni = SZE[0]-i-1
			if(state.location[0] == ni and state.location[1] == j and state.has_passenger == 1):
				print("T",end = "")   #Taxi has passenger
			elif(state.location[0] == ni and state.location[1] == j and state.has_passenger == 0):
				print("t",end = "")	  #Empty Taxi...
			elif((ni == state.pickup[0]) and (j == state.pickup[1])):
				print("S",end = "")
			elif(ni == depots[p.drop][0] and j == depots[p.drop][1]):
				print("D",end = "")
			else:
				print("0",end = "")
		print("")
	print("")

def display_2(state,p):
	time.sleep(1)
	system("clear")
	for i in range(2*SZE[0]+1):
		for j in range(2*SZE[1]+1):
			if(i%2 == 0):
				if(j%2 == 0):
					print(" ",end = "")
				else:
					if(i == 0):
						print("~",end = "")
					else:
						ni = SZE[0]-(i-1)//2-1
						nj = (j-1)//2
						if(walls['S'][ni][nj] == 1):
							print("~",end = "")
						else:	
							print(" ",end = "")
			else:
				if(j%2 == 0):
					if(j == 0):
						print("|",end = "")
					else:
						ni = SZE[0]-(i-1)//2-1
						nj = (j-1)//2
						if(walls['E'][ni][nj] == 1):
							print("|",end = "")
						else:	
							print(" ",end = "")
				else:
					ni = SZE[0]-i//2-1
					nj = j//2
					if(state.location[0] == ni and state.location[1] == nj and state.has_passenger == 1):
						print("T",end = "")   #Taxi has passenger
					elif(state.location[0] == ni and state.location[1] == nj and state.has_passenger == 0):
						print("t",end = "")	  #Empty Taxi...
					elif((ni == state.pickup[0]) and (nj == state.pickup[1])):
						print("S",end = "")
					elif(ni == depots[p.drop][0] and nj == depots[p.drop][1]):
						print("D",end = "")
					else:
						print("0",end = "")
		print("")
	print("")

def value_iter(p,eps,discount_factor):

	value1 = np.zeros((SZE[0],SZE[1],SZE[0],SZE[1],2))
	value2 = np.zeros((SZE[0],SZE[1],SZE[0],SZE[1],2))

	achieved_eps = Inf

	iterations = 0
	while(achieved_eps > eps):
		achieved_eps = 0

		for i in range(SZE[0]):
			for j in range(SZE[1]):
				for k in range(SZE[0]):
					for l in range(SZE[1]):
						for passenger in range(2):
							if(passenger == 1):
								if(i!=k or j!=l):
									continue
							st = state(np.array([i,j]),np.array([k,l]),passenger)
							maxx = -Inf

							for a in actions:
								val = 0
								d = p.next_state(st,a)

								for s2 in (d.keys()):
									val += d[s2]*(p.reward_model(st,a,s2) + discount_factor*valueFunction(value1,s2))
								maxx = max(val,maxx)
							
							value2[st.location[0],st.location[1],st.pickup[0],st.pickup[1],st.has_passenger] = maxx
							achieved_eps = max(achieved_eps,abs(maxx-value1[st.location[0],st.location[1],st.pickup[0],st.pickup[1],st.has_passenger]))
		
		value1 = value2.copy()
		iterations += 1
		#print_value(value1)
		#print(achieved_eps)

	print("The number of iterations taken are : "+str(iterations))
	return extract_policy(value2,p,discount_factor),iterations

def extract_policy(value,p,discount_factor):

	policy = np.empty((SZE[0],SZE[1],SZE[0],SZE[1],2),dtype = 'object')
	for i in range(SZE[0]):
		for j in range(SZE[1]):
			for k in range(SZE[0]):
				for l in range(SZE[1]):
					for passenger in range(2):
						st = state(np.array([i,j]),np.array([k,l]),passenger)
						maxx = -Inf

						for a in actions:
							val = 0
							d = p.next_state(st,a)
							for s2 in (d.keys()):
								val += d[s2]*(p.reward_model(st,a,s2) + discount_factor*valueFunction(value,s2))
							if(maxx<val):
								policy[i][j][k][l][passenger] = a
								#print(a)
								#print(policy[i][j][passenger])
								maxx = val
	return policy

def extract_value(policy,p,eps,discount_factor):
	value1 = np.zeros((SZE[0],SZE[1],SZE[0],SZE[1],2))
	value2 = np.zeros((SZE[0],SZE[1],SZE[0],SZE[1],2))

	achieved_eps = Inf

	while(achieved_eps > eps):
		achieved_eps = 0

		for i in range(SZE[0]):
			for j in range(SZE[1]):
				for k in range(SZE[0]):
					for l in range(SZE[1]):
						for passenger in range(2):
							if(passenger==1):
								if(i != k or j != l):
									continue
							st = state(np.array([i,j]),np.array([k,l]),passenger)
							val = 0
							d = p.next_state(st,policy[i][j][k][l][passenger])

							for s2 in (d.keys()):
								val += d[s2]*(p.reward_model(st,policy[i][j][k][l][passenger],s2) + discount_factor*valueFunction(value1,s2))
												
							value2[st.location[0],st.location[1],st.pickup[0],st.pickup[1],st.has_passenger] = val
							achieved_eps = max(achieved_eps,abs(val-value1[st.location[0],st.location[1],st.pickup[0],st.pickup[1],st.has_passenger]))
		
		value1 = value2.copy()
	return value1

def extract_value_linear_algebra(policy,p,eps,discount_factor):				#can further prune the states that are not possible..
	value = np.zeros((SZE[0],SZE[1],SZE[0],SZE[1],2))
	A = []
	b = []
	for i in range(SZE[0]):
			for j in range(SZE[1]):
				for k in range(SZE[0]):
					for l in range(SZE[1]):
						for passenger in range(2):
							st = state(np.array([i,j]),np.array([k,l]),passenger)
							d = p.next_state(st,policy[i][j][k][l][passenger])
							temp = np.zeros(SZE[0]*SZE[1]*SZE[0]*SZE[1]*2)
							temp[i*SZE[1]*SZE[0]*SZE[1]*2 + j*SZE[0]*SZE[1]*2+k*SZE[1]*2+l*2+passenger] = 1
							r = 0
							for s in d:
								t = d[s]
								temp[s.location[0]*SZE[1]*SZE[0]*SZE[1]*2 + s.location[1]*SZE[0]*SZE[1]*2+s.pickup[0]*SZE[1]*2+s.pickup[1]*2+s.has_passenger] -= discount_factor*t
								r += t*p.reward_model(st,policy[i][j][k][l][passenger],s)
							A.append(temp)
							b.append(r)
	ans = np.linalg.solve(np.array(A), np.array(b))
	for i in range(SZE[0]):
			for j in range(SZE[1]):
				for k in range(SZE[0]):
					for l in range(SZE[1]):
						for passenger in range(2):
							value[i][j][k][l][passenger] = ans[i*SZE[1]*SZE[0]*SZE[1]*2 + j*SZE[0]*SZE[1]*2+k*SZE[1]*2+l*2+passenger]
	return value

def policy_iter(p,eps,discount_factor):

	policy1 = np.empty((SZE[0],SZE[1],SZE[0],SZE[1],2),dtype = 'object')
	policy2 = np.empty((SZE[0],SZE[1],SZE[0],SZE[1],2),dtype = 'object')
	changed = True

	utilites = []
	#Initialize policy...
	for i in range(SZE[0]):
			for j in range(SZE[1]):
				for k in range(SZE[0]):
					for l in range(SZE[1]):
						for passenger in range(2):
							policy1[i][j][k][l][passenger] = random.choice(('N','E','W','S'))

	iterations = 0
	while(changed):
		#value = extract_value_linear_algebra(policy1,p,eps,discount_factor)
		value = extract_value(policy1,p,eps,discount_factor)
		#print(value1)
		#value = value2
		#print(value2-value1)
		utilites.append(value)
		policy2 = extract_policy(value,p,discount_factor)
		if((policy1 != policy2).any()):
			changed = True
		else:
			changed = False
		policy1 = policy2.copy()
		iterations += 1
	
	policy_loss = []
	for utility in utilites:
		policy_loss.append(np.max(abs(utility - utilites[-1])))

	#"""
	plt.plot(range(1,iterations+1),policy_loss)
	plt.xlabel("num of iterations")
	plt.ylabel("policy_loss")
	plt.title("Discount_factor = "+str(discount_factor))
	plt.show()
	#"""

	print("The number of iterations taken are : "+str(iterations))
	#value1 = extract_value_linear_algebra(policy1,p,eps,discount_factor)
	#value2 = extract_value(policy1,p,eps,discount_factor)
	#print(value2[2][2])
	#print(value1[2][2] - value2[2][2])
	#print(value2[2][2])
	return policy1

def q_learning(p,alpha,discount_factor,epsilon,exponential_decay = False):

	MAX_EPISODES = 2000 
	#MAX_EPISODES = 10000
	MAX_ITERATIONS = 500

	episode = 1
	Q = np.zeros((SZE[0],SZE[1],SZE[0],SZE[1],2,6)) # 6 actions
	initial_start = p.start

	reward_iterations = []
	while(episode<MAX_EPISODES):

		iterations = 1
		passenger = random.randint(0,1)
		#print(passenger)
		curr_state = None
		if(passenger == 1):
			l = np.random.randint(SZE[0],size = (2,))
			curr_state = state(l,np.array([l[0],l[1]]),passenger)
		else:
			curr_state = state(np.random.randint(SZE[0],size = (2,)),np.random.randint(SZE[0],size = (2,)),passenger)	#Do prune cases for which pickup != location for passenger == 1
		p.start = curr_state

		discounted_rewards = 0
		thisFactor = 1
		while(iterations<MAX_ITERATIONS):

			this_eps = epsilon if not exponential_decay else (epsilon)/iterations

			if(random.random()<this_eps):
				this_action_num = random.randint(0,5)
			else:
				this_action_num = Q[curr_state.location[0]][curr_state.location[1]][curr_state.pickup[0]][curr_state.pickup[1]][curr_state.has_passenger].argmax()

			#print(this_action_num)
			next_state = p.ret_next_state(curr_state,actions[this_action_num])
			this_reward = p.reward_model(curr_state,actions[this_action_num],next_state)

			discounted_rewards += this_reward*thisFactor
			thisFactor *= discount_factor

			if(next_state.location[0]==-1 and next_state.location[1]==-1):
				Q[curr_state.location[0]][curr_state.location[1]][curr_state.pickup[0]][curr_state.pickup[1]][curr_state.has_passenger][this_action_num] += alpha*(this_reward - Q[curr_state.location[0]][curr_state.location[1]][curr_state.pickup[0]][curr_state.pickup[1]][curr_state.has_passenger][this_action_num])
				break
			else:
				Q[curr_state.location[0]][curr_state.location[1]][curr_state.pickup[0]][curr_state.pickup[1]][curr_state.has_passenger][this_action_num] += alpha*(this_reward + discount_factor*np.max(Q[next_state.location[0]][next_state.location[1]][next_state.pickup[0]][next_state.pickup[1]][next_state.has_passenger]) - Q[curr_state.location[0]][curr_state.location[1]][curr_state.pickup[0]][curr_state.pickup[1]][curr_state.has_passenger][this_action_num])

			curr_state = state([next_state.location[0],next_state.location[1]],[next_state.pickup[0],next_state.pickup[1]],next_state.has_passenger)
			iterations += 1

		reward_iterations.append(discounted_rewards)
		#print(iterations)
		episode+=1

	#"""
	plt.plot(reward_iterations)
	plt.xlabel("Number of iterations")
	plt.ylabel("Total reward in episode")
	if exponential_decay:
		plt.title("Q-learning with exponential_decay")
	else:
		plt.title("Q-learning eps = "+str(epsilon)+", alpha = "+str(alpha))
	plt.show()
	#"""

	policy = np.empty((SZE[0],SZE[1],SZE[0],SZE[1],2),dtype = 'object')
	for i in range(SZE[0]):
			for j in range(SZE[1]):
				for k in range(SZE[0]):
					for l in range(SZE[1]):
						for passenger in range(2):
							policy[i][j][k][l][passenger] = actions[Q[i][j][k][l][passenger].argmax()]

	p.start = initial_start
	return policy

def sarsa_learning(p,alpha,discount_factor,epsilon,exponential_decay = False):

	MAX_EPISODES = 2000
	MAX_ITERATIONS = 500

	episode = 1
	Q = np.zeros((SZE[0],SZE[1],SZE[0],SZE[1],2,6)) # 6 actions
	initial_start = p.start

	reward_iterations = []
	while(episode<MAX_EPISODES):

		iterations = 1
		passenger = random.randint(0,1)
		#print(passenger)
		curr_state = None
		if(passenger == 1):
			l = np.random.randint(SZE[0],size = (2,))
			curr_state = state(l,np.array([l[0],l[1]]),passenger)
		else:
			curr_state = state(np.random.randint(SZE[0],size = (2,)),np.random.randint(SZE[0],size = (2,)),passenger)
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
				this_action_num = Q[curr_state.location[0]][curr_state.location[1]][curr_state.pickup[0]][curr_state.pickup[1]][curr_state.has_passenger].argmax()

			#print(this_action_num)
			next_state = p.ret_next_state(curr_state,actions[this_action_num])
			this_reward = p.reward_model(curr_state,actions[this_action_num],next_state)

			discounted_rewards += this_reward*thisFactor
			thisFactor *= discount_factor

			if(random.random()<this_eps):
				next_action_num = random.randint(0,5)
			else:
				next_action_num = Q[next_state.location[0]][next_state.location[1]][next_state.pickup[0]][next_state.pickup[1]][next_state.has_passenger].argmax()

			if(next_state.location[0]==-1 and next_state.location[1]==-1):
				Q[curr_state.location[0]][curr_state.location[1]][curr_state.pickup[0]][curr_state.pickup[1]][curr_state.has_passenger][this_action_num] += alpha*(this_reward - Q[curr_state.location[0]][curr_state.location[1]][curr_state.pickup[0]][curr_state.pickup[1]][curr_state.has_passenger][this_action_num])
				break
			else:
				Q[curr_state.location[0]][curr_state.location[1]][curr_state.pickup[0]][curr_state.pickup[1]][curr_state.has_passenger][this_action_num] += alpha*(this_reward + discount_factor*(Q[next_state.location[0]][next_state.location[1]][next_state.pickup[0]][next_state.pickup[1]][next_state.has_passenger][next_action_num]) - Q[curr_state.location[0]][curr_state.location[1]][curr_state.pickup[0]][curr_state.pickup[1]][curr_state.has_passenger][this_action_num])

			curr_state = state([next_state.location[0],next_state.location[1]],[next_state.pickup[0],next_state.pickup[1]],next_state.has_passenger)
			iterations += 1

		#print(iterations)
		reward_iterations.append(discounted_rewards)
		episode+=1

	#"""
	plt.plot(reward_iterations)
	plt.xlabel("Number of iterations")
	plt.ylabel("Total reward in episode")
	if exponential_decay:
		plt.title("SARSA-learning with exponential_decay")
	else:
		plt.title("SARSA-learning")
	plt.show()
	#"""

	policy = np.empty((SZE[0],SZE[1],SZE[0],SZE[1],2),dtype = 'object')
	for i in range(SZE[0]):
			for j in range(SZE[1]):
				for k in range(SZE[0]):
					for l in range(SZE[1]):
						for passenger in range(2):
							policy[i][j][k][l][passenger] = actions[Q[i][j][k][l][passenger].argmax()]

	p.start = initial_start
	return policy

def returnDisRewards(policy,p,discount_factor):			# Can also be used for running....simulation...
	TOTAL_RUNS = 5
	TOTAL_ITERATIONS = 500
	curr_run = 0
	totalRewards = 0

	initial_start = p.start

	while(curr_run<TOTAL_RUNS):
		print('New simulation episode--------------------')
		curr_state = state(np.random.randint(SZE[0],size = (2,)),np.random.randint(SZE[0],size = (2,)))
		#curr_state = state(depots['R'],depots['Y'])   #For A.2.c
		p.start = curr_state
		iterations = 0
		thisFactor = 1

		while(iterations<TOTAL_ITERATIONS):
			display_2(curr_state,p)
			thisAction = policy[curr_state.location[0]][curr_state.location[1]][curr_state.pickup[0]][curr_state.pickup[1]][curr_state.has_passenger]
			next_state = p.ret_next_state(curr_state,thisAction)
			this_reward = p.reward_model(curr_state,thisAction,next_state)

			# For A.2.c 
			#print(str(iterations) + ": ("+str(curr_state.location[0])+","+str(curr_state.location[1])+")-->"+thisAction)

			totalRewards += thisFactor*this_reward
			curr_state = state([next_state.location[0],next_state.location[1]],[next_state.pickup[0],next_state.pickup[1]],next_state.has_passenger)
			if(curr_state.location[0]==-1 and curr_state.location[1]==-1):
				break
			thisFactor *= discount_factor
			iterations += 1

		curr_run += 1

	p.start = initial_start
	return totalRewards/TOTAL_RUNS

def biggerDomain():
	global SZE
	global depots
	global walls

	previous_SZE = SZE.copy()
	previous_depots = depots.copy()
	previous_walls = walls.copy()

	SZE = [10,10]
	depots = {'R':np.array([9,0]),'Y':np.array([1,0]),'W':np.array([6,3]),'B':np.array([0,4]),'G':np.array([9,5]),'M':np.array([5,6]),'C':np.array([9,8]),'P':np.array([0,9])}
	walls = {}

	walls['N'] = np.array([[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[1,1,1,1,1,1,1,1,1,1]])
	walls['S'] = np.array([[1,1,1,1,1,1,1,1,1,1],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]])
	walls['E'] = np.array([[1,0,0,1,0,0,0,1,0,1],[1,0,0,1,0,0,0,1,0,1],[1,0,0,1,0,0,0,1,0,1],[1,0,0,1,0,0,0,1,0,1],[0,0,0,0,0,1,0,0,0,1],[0,0,0,0,0,1,0,0,0,1],[0,0,1,0,0,1,0,1,0,1],[0,0,1,0,0,1,0,1,0,1],[0,0,1,0,0,0,0,1,0,1],[0,0,1,0,0,0,0,1,0,1]])
	walls['W'] = np.array([[1,1,0,0,1,0,0,0,1,0],[1,1,0,0,1,0,0,0,1,0],[1,1,0,0,1,0,0,0,1,0],[1,1,0,0,1,0,0,0,1,0],[1,0,0,0,0,0,1,0,0,0],[1,0,0,0,0,0,1,0,0,0],[1,0,0,1,0,0,1,0,1,0],[1,0,0,1,0,0,1,0,1,0],[1,0,0,1,0,0,0,0,1,0],[1,0,0,1,0,0,0,0,1,0]])

	drop = 'R'

	start = np.array([7,9])
	p = parta1(start,drop)

	this_policy = q_learning(p,0.25,0.99,0.1)
	print(returnDisRewards(this_policy,p,0.99))

	SZE = previous_SZE.copy()
	depots = previous_depots.copy()
	walls = previous_walls.copy()


def partA2_b(this_discount_factor):
	epsilon_list = [0.005,0.01,0.02,0.04,0.08,0.16]
	iterations = []

	drop = 'Y'
	start = np.array([4,4])
	p = parta1(start,drop)

	for epsilon in epsilon_list:
		_,thisIter = value_iter(p,epsilon,this_discount_factor)
		iterations.append(thisIter)

	plt.plot(epsilon_list,iterations)
	plt.xlabel("Epsilons")
	plt.ylabel("Number of iterations")
	plt.title("Discount_factor = "+str(this_discount_factor))
	plt.show()


if __name__ == "__main__":

	drop = 'R'
	start = np.array([4,4])
	p = parta1(start,drop)

	#this_policy = q_learning(p,0.5,0.99,0.1)
	#this_policy = q_learning(p,0.25,0.99,0.1,True)
	#this_policy = sarsa_learning(p,0.25,0.99,0.1)
	#this_policy = sarsa_learning(p,0.25,0.99,0.1,True)
	#print(returnDisRewards(this_policy,p,0.99))

	#this_policy,_ = value_iter(p,0.01,0.9)
	#print(returnDisRewards(this_policy,p,0.9))
	this_policy = q_learning(p,0.25,0.99,0.1)
	a = input('Press key to enter simulation....')
	print(returnDisRewards(this_policy,p,0.9))
	#partA2_b(0.1)


	#print('For larger game....')
	#biggerDomain()
	