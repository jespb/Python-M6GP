from .Individual import Individual
from .Node import Node

# 
# By using this file, you are agreeing to this product's EULA
#
# This product can be obtained in https://github.com/jespb/Python-M6GP
#
# Copyright ©2023-2024 J. E. Batista
#


def tournament(rng, population,n):
	'''
	Selects "n" Individuals from the population and return a 
	single Individual.

	Parameters:
	population (list): A list of Individuals, sorted from best to worse.
	'''
	candidates = [rng.randint(0,len(population)-1) for i in range(n)]
	return population[min(candidates)]

def doubleTournament(rng, population, n):
	candidate = [ tournament(rng, population, n) for i in range(n)]
	selected = candidate[0]
	for i in range(1,n):
		if candidate[i].crowdingDistance > selected.crowdingDistance:
			selected= candidate[i]
	return selected


def splitIntoFrontiers(population):
	frontiers = []
	population = population[:]

	while len(population) > 0:
		pf = getParetoFrontier(population)
		for ind in pf:
			population.pop( population.index(ind) )
		frontiers.append(pf)

	return frontiers


def getParetoFrontier(population):
	#print("  >>> Calculating Pareto Frontier")
	frontier = []
	for ind in population:

		isDominated = False
		i = 0
		while i < len(population) and not isDominated:
			isDominated =  population[i] > ind
			i = i+1
		if not isDominated:
			frontier.append(ind)
	return frontier


def getOffspring(rng, population, tournament_size):
	'''
	Genetic Operator: Selects a genetic operator and returns a list with the 
	offspring Individuals. The crossover GOs return two Individuals and the
	mutation GO returns one individual. Individuals over the LIMIT_DEPTH are 
	then excluded, making it possible for this method to return an empty list.

	Parameters:
	population (list): A list of Individuals, sorted from best to worse.
	'''
	#print("  >>> Breeding")

	isCross = rng.random()<0.5
	desc = None

	availableXO = [0,1]
	availableMT = [0,1,2] 

	if isCross:
		whichXO = availableXO[ rng.randint(0,len(availableXO)-1 ) ]
		if whichXO == 0:
			desc = STXO(rng, population, tournament_size)
		elif whichXO == 1:
			desc = M5XO(rng, population, tournament_size)
	else:
		whichMut = availableMT[ rng.randint(0,len(availableMT)-1 ) ]
		if whichMut == 0:
			desc = STMUT(rng, population, tournament_size)
		elif whichMut == 1:
			desc = M5ADD(rng, population, tournament_size)
		elif whichMut == 2:
			desc = M5REM(rng, population, tournament_size)
	return desc


def discardIndividualWithInvalidSize(population, min_dim, max_dim, depth_limit):
	ret = []
	for ind in population:
		dimensions = ind.getNumberOfDimensions()
		depth = ind.getDepth() if dimensions > 0 else -1
		if depth <= depth_limit and dimensions >= min_dim and dimensions <= max_dim:
			ret.append(ind)
	return ret


def STXO(rng, population, tournament_size):
	'''
	Randomly selects one node from each of two individuals; swaps the node and
	sub-nodes; and returns the two new Individuals as the offspring.

	Parameters:
	population (list): A list of Individuals, sorted from best to worse.
	'''
	ind1 = doubleTournament(rng, population, tournament_size)
	ind2 = doubleTournament(rng, population, tournament_size)

	d1 = ind1.getDimensions()
	d2 = ind2.getDimensions()

	r1 = rng.randint(0,len(d1)-1)
	r2 = rng.randint(0,len(d2)-1)

	n1 = d1[r1].getRandomNode(rng)
	n2 = d2[r2].getRandomNode(rng)

	n1.swap(n2)

	ret = []
	for d in [d1,d2]:
		i = Individual(ind1.operators, ind1.terminals, ind1.max_depth, ind1.model_name, ind1.fitnesses)
		i.copy(d)
		ret.append(i)
	return ret

def M5XO(rng, population, tournament_size):
	'''
	Randomly selects one dimension from each of two individuals; swaps the 
	dimensions; and returns the two new Individuals as the offspring.

	Parameters:
	population (list): A list of Individuals, sorted from best to worse.
	'''
	ind1 = doubleTournament(rng, population, tournament_size)
	ind2 = doubleTournament(rng, population, tournament_size)

	d1 = ind1.getDimensions()
	d2 = ind2.getDimensions()

	r1 = rng.randint(0,len(d1)-1)
	r2 = rng.randint(0,len(d2)-1)

	d1.append(d2[r2])
	d2.append(d1[r1])
	d1.pop(r1)
	d2.pop(r2)

	ret = []
	for d in [d1,d2]:
		i = Individual(ind1.operators, ind1.terminals, ind1.max_depth, ind1.model_name, ind1.fitnesses)
		i.copy(d)
		ret.append(i)
	return ret

def STMUT(rng, population, tournament_size):
	'''
	Randomly selects one node from a single individual; swaps the node with a 
	new, node generated using Grow; and returns the new Individual as the offspring.

	Parameters:
	population (list): A list of Individuals, sorted from best to worse.
	'''
	ind1 = doubleTournament(rng, population, tournament_size)
	d1 = ind1.getDimensions()
	r1 = rng.randint(0,len(d1)-1)
	n1 = d1[r1].getRandomNode(rng)
	n = Node()
	n.create(rng, ind1.operators, ind1.terminals, ind1.max_depth)
	n1.swap(n)


	ret = []
	i = Individual(ind1.operators, ind1.terminals, ind1.max_depth, ind1.model_name, ind1.fitnesses)
	i.copy(d1)
	ret.append(i)
	return ret

def M5ADD(rng, population, tournament_size):
	'''
	Randomly generates a new node using Grow; this node is added to the list of
	dimensions; the new Individual is returned as the offspring.

	Parameters:
	population (list): A list of Individuals, sorted from best to worse.
	'''
	ind1 = doubleTournament(rng, population, tournament_size)
	ret = []

	d1 = ind1.getDimensions()
	n = Node()
	n.create(rng, ind1.operators, ind1.terminals, ind1.max_depth)
	d1.append(n)

	i = Individual(ind1.operators, ind1.terminals, ind1.max_depth, ind1.model_name, ind1.fitnesses)
	i.copy(d1)
	ret.append(i)

	return ret

def M5REM(rng, population, tournament_size):
	'''
	Randomly selects one dimensions from a single individual; that dimensions is
	removed; the new Individual is returned as the offspring.

	Parameters:
	population (list): A list of Individuals, sorted from best to worse.
	'''
	ind1 = doubleTournament(rng, population, tournament_size)
	ret = []

	d1 = ind1.getDimensions()
	r1 = rng.randint(0,len(d1)-1)
	d1.pop(r1)
		
	i = Individual(ind1.operators, ind1.terminals, ind1.max_depth, ind1.model_name, ind1.fitnesses)
	i.copy(d1)
	ret.append(i)
	
	return ret
