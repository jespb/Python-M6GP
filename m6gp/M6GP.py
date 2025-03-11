from .Individual import Individual
from .GeneticOperators import splitIntoFrontiers, getParetoFrontier, getOffspring, discardIndividualWithInvalidSize
import multiprocessing as mp
import time
import matplotlib.pyplot as plt


from .MahalanobisDistanceClassifier import MahalanobisDistanceClassifier

from random import Random

# 
# By using this file, you are agreeing to this product's EULA
#
# This product can be obtained in https://github.com/jespb/Python-M6GP
#
# Copyright ©2023-2024 J. E. Batista
#


class ClassifierNotTrainedError(Exception):
    """ You tried to use the classifier before training it. """

    def __init__(self, expression, message = ""):
        self.expression = expression
        self.message = message


class M6GP:

	## __INIT__ arguments
	operators = None
	max_initial_depth = None
	population_size = None
	threads: int = None
	random_state: int = 42
	rng = None # random number generator

	max_depth: int = None
	max_generation: int = None
	dim_min: int = None
	dim_max: int = None

	model_class = None 
	fitnesses = None

	verbose = None


	## FIT arguments
	terminals = None

	elite = None
	population = None
	currentGeneration = 0
	bestIndividual: Individual = None

	trainingAccuracyOverTime = None
	testAccuracyOverTime = None
	trainingWaFOverTime = None
	testWaFOverTime = None
	trainingKappaOverTime = None
	testKappaOverTime = None
	trainingMSEOverTime = None
	testMSEOverTime = None
	sizeOverTime = None
	dimensionsOverTime = None
	generationTimes = None
	solutionsPerGeneration = None



	def checkIfTrained(self):
		if self.population == None:
			raise ClassifierNotTrainedError("The classifier must be trained using the fit(Tr_X, Tr_Y) method before being used.")



	def __init__(self, operators=[("+",2),("-",2),("*",2),("/",2)], max_initial_depth = 6, 
		population_size = 500, max_generation = 100, max_depth = 17, tournament_size = 5,
		dim_min = 1, dim_max = 9999, threads=1, random_state = 42, verbose = True, 
		model_class=None, fitnesses=["Accuracy","Size"]):

		if sum( [0 if op in [("+",2),("-",2),("*",2),("/",2)] else 0 for op in operators ] ) > 0:
			print( "[Warning] Some of the following operators may not be supported:", operators)

		self.operators = operators

		self.max_initial_depth = max_initial_depth
		self.population_size = population_size
		self.threads = max(1, threads)
		self.random_state = random_state
		self.rng = Random(random_state)
		self.tournament_size = tournament_size

		self.max_depth = max_depth
		self.max_generation = max_generation
		self.dim_min = max(1, dim_min)
		self.dim_max = max(1, dim_max)

		self.model_class = model_class
		if self.model_class is None:
			self.model_class = MahalanobisDistanceClassifier()

		self.fitnesses = fitnesses

		self.verbose = verbose





	def __str__(self):
		self.checkIfTrained()
		return str(self.getBestIndividual())
		



	def getCurrentGeneration(self):
		return self.currentGeneration


	def getBestIndividual(self):
		'''
		Returns the final M6GP model.
		'''
		self.checkIfTrained()

		return self.bestIndividual

	def getAccuracyOverTime(self):
		'''
		Returns the training and test accuracy of the best model in each generation.
		'''
		self.checkIfTrained()

		return [self.trainingAccuracyOverTime, self.testAccuracyOverTime]

	def getWaFOverTime(self):
		'''
		Returns the training and test WAF of the best model in each generation.
		'''
		self.checkIfTrained()

		return [self.trainingWaFOverTime, self.testWaFOverTime]

	def getKappaOverTime(self):
		'''
		Returns the training and test kappa values of the best model in each generation.
		'''
		self.checkIfTrained()

		return [self.trainingKappaOverTime, self.testKappaOverTime]

	def getMSEOverTime(self):
		'''
		Returns the training and test mean squared error values of the best model in each generation.
		'''
		self.checkIfTrained()

		return [self.trainingMSEOverTime, self.testMSEOverTime]

	def getSizesOverTime(self):
		'''
		Returns the size and number of dimensions of the best model in each generation.
		'''
		self.checkIfTrained()

		return [self.sizeOverTime, self.dimensionsOverTime]

	def getGenerationTimes(self):
		'''
		Returns the time spent in each generation.
		'''
		self.checkIfTrained()

		return self.generationTimes










	def fit(self,Tr_x, Tr_y, Te_x = None, Te_y = None):
		if self.verbose:
			print("  > Parameters")
			print("    > Random State:       "+str(self.random_state))
			print("    > Operators:          "+str(self.operators))
			print("    > Population Size:    "+str(self.population_size))
			print("    > Max Generation:     "+str(self.max_generation))
			print("    > Max Initial Depth:  "+str(self.max_initial_depth))
			print("    > Max Depth:          "+str(self.max_depth))
			print("    > Minimum Dimensions: "+str(self.dim_min))
			print("    > Maximum Dimensions: "+str(self.dim_max))
			print("    > Wrapped Model:      "+self.model_class.__class__.__name__)
			print("    > Fitnesses:          "+" - ".join(self.fitnesses))
			print("    > Threads:            "+str(self.threads))
			print()

		self.Tr_x = Tr_x
		self.Tr_y = Tr_y
		self.Te_x = Te_x
		self.Te_y = Te_y
		self.terminals = list(Tr_x.columns)


		self.elite = []
		self.population = []

		while len(self.population) < self.population_size:
			ind = Individual(self.operators, self.terminals, self.max_depth, self.model_class, self.fitnesses)
			ind.create(self.rng, n_dims = self.dim_min)
			self.population.append(ind)

		self.bestIndividual = self.population[0]
		self.bestIndividual.fit(self.Tr_x, self.Tr_y)

		if not self.Te_x is None:
			self.trainingAccuracyOverTime = []
			self.testAccuracyOverTime = []
			self.trainingWaFOverTime = []
			self.testWaFOverTime = []
			self.trainingKappaOverTime = []
			self.testKappaOverTime = []
			self.trainingMSEOverTime = []
			self.testMSEOverTime = []
			self.sizeOverTime = []
			self.dimensionsOverTime = []
			self.generationTimes = []
			self.solutionsPerGeneration = []



		'''
		Training loop for the algorithm.
		'''
		if self.verbose:
			print("  > Running log:\n")

		while self.currentGeneration < self.max_generation:
			if not self.stoppingCriteria():
				t1 = time.time()
				self.nextGeneration()
				t2 = time.time()
				duration = t2-t1
			else:
				duration = 0
			self.currentGeneration += 1
			
			if not self.Te_x is None:
				if self.fitnesses[0] in ["Accuracy", "2FOLD", "WAF"]:
					self.trainingAccuracyOverTime.append(self.bestIndividual.getAccuracy(self.Tr_x, self.Tr_y, pred="Tr"))
					self.testAccuracyOverTime.append(self.bestIndividual.getAccuracy(self.Te_x, self.Te_y, pred="Te"))
					self.trainingWaFOverTime.append(self.bestIndividual.getWaF(self.Tr_x, self.Tr_y, pred="Tr"))
					self.testWaFOverTime.append(self.bestIndividual.getWaF(self.Te_x, self.Te_y, pred="Te"))
					self.trainingKappaOverTime.append(self.bestIndividual.getKappa(self.Tr_x, self.Tr_y, pred="Tr"))
					self.testKappaOverTime.append(self.bestIndividual.getKappa(self.Te_x, self.Te_y, pred="Te"))
					self.trainingMSEOverTime.append(0)
					self.testMSEOverTime.append(0)
				elif self.fitnesses[0] in ["MSE"]:
					self.trainingAccuracyOverTime.append(0)
					self.testAccuracyOverTime.append(0)
					self.trainingWaFOverTime.append(0)
					self.testWaFOverTime.append(0)
					self.trainingKappaOverTime.append(0)
					self.testKappaOverTime.append(0)
					self.trainingMSEOverTime.append(self.bestIndividual.getMSE(self.Tr_x, self.Tr_y, pred="Tr"))
					self.testMSEOverTime.append(self.bestIndividual.getMSE(self.Te_x, self.Te_y, pred="Te"))
				self.sizeOverTime.append(self.bestIndividual.getSize())
				self.dimensionsOverTime.append(self.bestIndividual.getNumberOfDimensions())
				self.generationTimes.append(duration)
				self.solutionsPerGeneration.append( [ind.fitness for ind in self.old_population] )


		# prun the final individual
		#self.getBestIndividual().prun(min_dim = self.dim_min, simp=True)




	def stoppingCriteria(self):
		'''
		Returns True if the stopping criteria was reached.
		'''
		genLimit = self.currentGeneration >= self.max_generation
		#perfectTraining = self.bestIndividual.getFitness() == 1
		
		return genLimit  #or perfectTraining




	def nextGeneration(self):
		'''
		Generation algorithm: the population is sorted; the best individual is pruned;
		the elite is selected; and the offspring are created.
		'''
		begin = time.time()

		# Calculates the accuracy of the population using multiprocessing

		#print("  >> Calculating fitness... ")
		if self.threads > 1:
			#print("Using multiprocessing")
			with mp.Pool(processes= self.threads) as pool:
				results = pool.map(fitIndividuals, [(ind, self.Tr_x, self.Tr_y) for ind in self.population] )
				for i in range(len(self.population)):
					self.population[i].trainingPredictions = results[i][0]
					self.population[i].fitness = results[i][1]
					self.population[i].training_X = self.Tr_x
					self.population[i].training_Y = self.Tr_y
					self.population[i].model = results[i][2]
		else:
			#print("Using a single thread")
			[ ind.fit(self.Tr_x, self.Tr_y) for ind in self.population]
			[ ind.getFitnesses() for ind in self.population ]

		#print("  >> Adding elite to population...")
		self.population.extend(self.elite)

		#print("  >> Updating individual with best performance...")
		for ind in self.population:
			if ind.fitness[0] > self.bestIndividual.fitness[0]:
				self.bestIndividual = ind
		if self.population[0] > self.bestIndividual:
			self.bestIndividual = self.population[0]
			#self.bestIndividual.fit(self.Tr_x, self.Tr_y)
			#self.bestIndividual.prun(min_dim = self.dim_min)


		#print("  >> Calculating number of dominated solutions...")
		[ind.getNumberOfDominatedSolutions(self.population) for ind in self.population]


		#print("  >> Calculating crownding distance...")
		frontiers = splitIntoFrontiers(self.population)
		for f in frontiers:
			for ind in f:
				ind.crowdingDistance = 0
			for objective in range(len(f[0].fitnesses)):
				f.sort(key= lambda x:x.fitness[objective], reverse=True)
				fmin = f[-1].fitness[objective]
				fmax = f[0].fitness[objective]
				if fmax-fmin>0.00001: # if this fitness is not constant
					f[0].crowdingDistance  = 99999
					f[-1].crowdingDistance = 99999
					for i in range(2, len(f)-1):
						f[i].crowdingDistance += \
							(f[i+1].fitness[objective]-f[i-1].fitness[objective])/(fmax-fmin)


		#print("  >> Sorting population using the number of dominated solutions...")
		self.population.sort(reverse=True)


		#print("  >> Generating Next Generation...")
		self.elite = getParetoFrontier(self.population)
		newPopulation = []

		while len(newPopulation) < self.population_size:
			offspring = getOffspring(self.rng, self.population, self.tournament_size)
			offspring = discardIndividualWithInvalidSize(offspring, self.dim_min, self.dim_max, self.max_depth)
			newPopulation.extend(offspring)
		self.old_population = self.population
		self.population = newPopulation[:self.population_size]

		end = time.time()

		#print("  >> Printing Progress...")
		if self.verbose and self.currentGeneration%1==0:
			if not self.Te_x is None:
				print("   > Gen #%2d:  Fitness: [ %s ] // Tr-Score: %.6f // Te-Score: %.6f  // Time: %.4f" % (self.currentGeneration, " - ".join("%.4f"%x for x in self.bestIndividual.getFitnesses()), self.bestIndividual.getTrainingMeasure(), self.bestIndividual.getTestMeasure(self.Te_x, self.Te_y), end- begin )  )
			else:
				print("   > Gen #%2d:  Fitness: [ %s ] // Tr-Score: %.6f // Time: %.4f" % (self.currentGeneration, " - ".join("%.4f"%x for x in self.bestIndividual.getFitnesses()), self.bestIndividual.getTrainingMeasure(), end- begin )  )
			







	def predict(self, dataset):
		'''
		Returns the predictions for the samples in a dataset.
		'''
		self.checkIfTrained()

		return self.getBestIndividual().predict(dataset)

		return "Population Not Trained" if self.bestIndividual == None else self.bestIndividual.predict(sample)


def fitIndividuals(a):
	ind,x,y = a
	ind.getFitnesses(x,y)

	ret = []
	if "FOLD" in ind.fitnesses:
		ret.append(None)
	else:
		ret.append(ind.getTrainingPredictions())
	ret.append(ind.getFitnesses())
	ret.append(ind.model)

	
	return ret 






