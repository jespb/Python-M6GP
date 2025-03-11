import pandas

from m6gp.M6GP import M6GP
from sys import argv
from Arguments import *
import os

from sklearn.model_selection import train_test_split

import numpy as np

import warnings

warnings.filterwarnings("ignore", category=FutureWarning,
                        message="From version 0.21, test_size will always complement",
                        module="sklearn")


# 
# By using this file, you are agreeing to this product's EULA
#
# This product can be obtained in https://github.com/jespb/Python-M6GP
#
# Copyright ©2023-2024 J. E. Batista
#



def openAndSplitDatasets(which,seed):
	if VERBOSE:
		print( "> Opening: ", which )

	# Open dataset
	ds = pandas.read_csv(DATASETS_DIR+which)

	# Read header
	class_header = ds.columns[-1]

	return train_test_split(ds.drop(columns=[class_header]), ds[class_header], 
		train_size=TRAIN_FRACTION, random_state=seed, 
		stratify = ds[class_header])


def run(r,dataset):
	if VERBOSE:
		print("> Starting run:")
		print("  > ID:", r)
		print("  > Dataset:            "+dataset)
		print()

	Tr_X, Te_X, Tr_Y, Te_Y = openAndSplitDatasets(dataset,r)

	# Train a model
	model = M6GP(OPERATORS, MAX_DEPTH, POPULATION_SIZE, MAX_GENERATION, 
		LIMIT_DEPTH, TOURNAMENT_SIZE, DIM_MIN, DIM_MAX, THREADS, r, VERBOSE, MODEL, FITNESSES)
	model.fit(Tr_X, Tr_Y, Te_X, Te_Y)


	# Obtain training results
	accuracy  = model.getAccuracyOverTime()
	waf       = model.getWaFOverTime()
	kappa     = model.getKappaOverTime()
	mse       = model.getMSEOverTime()
	sizes     = model.getSizesOverTime()
	model_str = str(model.getBestIndividual())
	times     = model.getGenerationTimes()
	
	tr_acc     = accuracy[0]
	te_acc     = accuracy[1]
	tr_waf     = waf[0]
	te_waf     = waf[1]
	tr_kappa   = kappa[0]
	te_kappa   = kappa[1]
	tr_mse     = mse[0]
	te_mse     = mse[1]
	size       = sizes[0]
	dimensions = sizes[1]

	if VERBOSE:
		print("> Ending run:")
		print("  > ID:", r)
		print("  > Dataset:", dataset)
		print("  > Final model:", model_str)
		print("  > Training accuracy:", tr_acc[-1])
		print("  > Test accuracy:", te_acc[-1])
		print()

	return (tr_acc,te_acc,
			tr_waf,te_waf,
			tr_kappa,te_kappa,
			tr_mse,te_mse,
			size,dimensions,
			times,
			model_str,
			model.solutionsPerGeneration)
			

def callM6GP():
	try:
		os.makedirs(OUTPUT_DIR)
	except:
		pass

	for dataset in DATASETS:
		outputFilename = OUTPUT_DIR+"m6gp_"+ dataset
		if not os.path.exists(outputFilename):
			results = []

			# Run the algorithm several times
			for r in range(RUNS):
				results.append(run(r,dataset))

				# Write output header
				file = open(outputFilename , "w")
				file.write("Attribute,Run,")
				for i in range(MAX_GENERATION):
					file.write(str(i)+",")
				file.write("\n")
			
				attributes= ["Training-Accuracy","Test-Accuracy",
							 "Training-WaF", "Test-WaF",
							 "Training-Kappa", "Test-Kappa",
							 "Training-MSE", "Test-MSE",
							 "Size","Dimensions",
							 "Time",	
							 "Final_Model"]

				# Write attributes with value over time
				for ai in range(len(attributes)-1):
					for i in range(len(results)):	
						file.write("\n"+attributes[ai]+","+str(i)+",")
						file.write( ",".join([str(val) for val in results[i][ai]]))
					file.write("\n")

				# Write the final models
				for i in range(len(results)):
					file.write("\n"+attributes[-1]+","+str(i)+",")
					file.write(results[i][-2])
				file.write("\n")


				## Writes a 4D matrix containing the values of each individual in each objective over all generations 
				## results[r2][-1] == [generation] [individual ID (sorted by first objective)] [objective ID]
				#file.write("\nObjective values: %s"%FITNESSES)
				#for r2 in range(len(results)):
				#	file.write("\n%d,%s"%(r2, str(results[r2][-1])))
				#file.write("\n")

				# Write some parameters
				file.write("\n\nParameters")
				file.write("\nOperators,"+str(OPERATORS))
				file.write("\nMax Initial Depth,"+str(MAX_DEPTH))
				file.write("\nPopulation Size,"+str(POPULATION_SIZE))
				file.write("\nMax Generation,"+str(MAX_GENERATION))
				file.write("\nDepth Limit,"+str(LIMIT_DEPTH))
				file.write("\nMinimum Dimensions,"+str(DIM_MIN))
				file.write("\nMaximum Dimensions,"+str(DIM_MAX))
				file.write("\nWrapped Model,"+MODEL_NAME)
				file.write("\nFitness Type,"+" - ".join(FITNESSES))
				file.write("\nThreads,"+str(THREADS))
				file.write("\nRandom State,"+str(list(range(RUNS))))
				file.write("\nDataset,"+dataset)




				file.close()
		else:
			print("Filename: " + outputFilename +" already exists.")


if __name__ == '__main__':
	callM6GP()
