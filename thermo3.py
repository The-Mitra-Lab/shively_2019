
import itertools
import numpy as np
import seaborn as sns
import pandas as pd

def shea_ackers(depsilon_list,position_list,cooperativity_energy,tau,H):
	num_sites = len(depsilon_list)
	state_list = list(itertools.product([0,1],repeat = num_sites))
	unnorm_exp_occ_partial_sum = 0
	Z_partial_sum = 0
	#loop over states e.g. [0,0,0], [0,0,1] etc
	for sigma_list in state_list:
		print "state = "+str(sigma_list)
		state_occupancy = sum(sigma_list)
		print "occupancy = "+str(state_occupancy)

		#compute the state weight for state sigma_list
		if state_occupancy == 0:  				#This is state [0,0,0,0,0...0]
			state_weight = 1     				#The weight is 1
		else:
			exponent_partial_sum = 0
			#loop over sigma_i's for the state  
			for i,sigmai in enumerate(sigma_list):
				exponent_partial_sum = exponent_partial_sum + (depsilon_list[i]*sigmai)
			exponent = (-1*exponent_partial_sum) - (state_occupancy - 1)*cooperativity_energy
			state_weight = tau**(state_occupancy)*np.exp(exponent)
		#I know the occupancy and state weight for this state, so add to the building sum
		print "weight = " + str(state_weight)
		#print
		#the state weight has been computed.  Now compute the contribution of this 
		#state to occupancy
		unnorm_exp_occ_partial_sum = unnorm_exp_occ_partial_sum + state_occupancy*state_weight
		#add state weight of this state to the parition function
		Z_partial_sum = Z_partial_sum + state_weight

	#I've gone through all of the states, so normalize the final expected occupancy
	#by the partition function

	###print "Z = "+str(Z_partial_sum)
	expected_occupancy = unnorm_exp_occ_partial_sum/Z_partial_sum
	###print "expected_occupancy = "+str(expected_occupancy)
	return(H*expected_occupancy)


def fast_shea_ackers(depsilon_list,H,cooperativity_energy,tau):
	num_sites = len(depsilon_list)
	unnorm_exp_occ_partial_sum = 0
	Z_partial_sum = 0

	#We will sum the state weights of all weights with the same occupancy at once, rather
	#than considering each state separately

	#convert depsilon_list to k_list
	K_list = [np.exp(-x) for x in depsilon_list]
	#print "K_list:" + str(K_list)

	#find Kc
	Kc = np.exp(-cooperativity_energy)
	S_list = list(K_list)
	for occupancy in range(0,len(depsilon_list)+1): #loop over occupancies from 0 to N
		if occupancy == 0:
			osummed_state_weight = 1
		else:
			osummed_state_weight = tau**(occupancy)*(Kc**(occupancy-1))*sum(S_list)
			#print "occupancy: "+str(occupancy)+" weight: "+str(osummed_state_weight)
			#print "occupancy: "+str(occupancy)+" S_list = "+str(S_list)
			S_list = multiply_ks(K_list,S_list)
			
		unnorm_exp_occ_partial_sum = unnorm_exp_occ_partial_sum + osummed_state_weight *occupancy
		Z_partial_sum = Z_partial_sum + osummed_state_weight
	#I've looped through all of the occupancies, so now normalize	
	expected_occupancy = unnorm_exp_occ_partial_sum/Z_partial_sum
	return(H*expected_occupancy)	



def multiply_ks(K_list,S_list):
	new_S_list = [0]*(len(S_list)-1)
	run_sum = 0
	for i in range(len(S_list)-1,0,-1):
		new_S_list[i-1] = K_list[i-1]*(run_sum + S_list[i])
		run_sum = run_sum + S_list[i]
	return(new_S_list)

def fast_shea_ackers_series(epsilon,hops,scaling_factor,cooperativity,tau):
	occ_list = []
	for idx,line in epsilon.iterrows():
		occ_list.append(fast_shea_ackers(list(epsilon.loc[idx]),scaling_factor,cooperativity,tau))
	occ_list = pd.DataFrame(occ_list)
	occ_hops = pd.concat([occ_list,hops],axis=1)
	occ_hops.columns = ["Occupancy","TPH"]
	sns.regplot(x=occ_hops["Occupancy"],y=occ_hops["TPH"],ci = None)
	print "Correlation = "+str(round(np.corrcoef(occ_hops["Occupancy"],occ_hops["TPH"])[0,1],2))
	return occ_hops


def fast_shea_ackers_series_log(epsilon,hops,scaling_factor,cooperativity,tau):
	occ_list = []
	for idx,line in epsilon.iterrows():
		occ_list.append(fast_shea_ackers(list(epsilon.loc[idx]),scaling_factor,cooperativity,tau))
	occ_list = pd.DataFrame(occ_list)
	occ_hops = pd.concat([occ_list,hops],axis=1)
	occ_hops.columns = ["Occupancy","TPH"]
	occ_hops["Occupancy"] = occ_hops["Occupancy"].apply(rmlog2)
	occ_hops["TPH"] = occ_hops["TPH"].apply(rmlog2)
	sns.regplot(x=occ_hops["Occupancy"],y=occ_hops["TPH"],ci = None)
	print "Correlation = "+str(round(np.corrcoef(occ_hops["Occupancy"],occ_hops["TPH"])[0,1],2))
	return occ_hops

def rmlog2(x):
	if x == 0:
		x = 1
	return np.log2(x)







