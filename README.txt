This is an easy-to-use, scikit-learn inspired version of the M6GP algorithm.


By using this file, you are agreeing to this product's EULA
This product can be obtained at https://github.com/jespb/Python-M6GP
Copyright ©2023-2025 J. E. Batista


This file contains information about the command and flags used in the stand-alone version of this implementation and an explanation of how to import, use, and edit this implementation.




This implementation of M6GP can be used in a stand-alone fashion using the following command and flags:

$ python Main_M6GP_standalone.py
	
    [-d datasets] 
        - This flag expects a set of csv dataset names separated by ";" (e.g., "a.csv;b.csv")
        - By default, the heart.csv dataset is used		

    [-dsdir dir] 
        - States the dataset directory. 
        - By default "datasets/" is used 
        - Use "-dsdir ./" for the root directory	

    [-md max_depth]
        - This flag expects an integer with the maximum initial depth for the trees;
        - By default, this value is set to 6.		

    [-mg max_generation]
        - This flag expects an integer with the maximum number of generations;
        - By default, this value is set to 100.

    [-odir dir] 
        - States the output directory. 
        - By default, "results/" is used 
        - Use "-odir ./" for the root directory
	
    [-op operators]
        - This flag excepts a set of operators and their number of arguments, separated by ";"
        - Allowed operators: +,2 ; -,2 ; *,2 ; /,2
        - By default, the used operators are the sum, subtraction, multiplication, and protected division: "+,2;-,2;*,2;/,2"	

    [-ps population_size]
        - This flag expects an integer with the size of the population;
        - By default, this value is set to 500.

    [-runs number_of_runs] 
        - This flag expects an integer with the number of runs to be made;
        - By default, this value is set to 30
	
    [-tf train_fraction]
        - This flag expects a float [0;1] with the fraction of the dataset to be used in training;
        - By default, this value is set to 0.70
	
    [-ts tournament_size]
        - This flag expects an integer with the tournament size;
        - By default, this value is set to 10.

    [-t number_of_threads]
        - This flag expects an integer with the number of threads to use while evaluating the population;
        - If the value is set to 1, the multiprocessing library will not be used 
        - By default, this value is set to 1.
	
    [-di minimum_number_of_dimension]
        - This flag expects an integer with the minimum number of dimensions in each individual;
        - This flag affects the number of dimensions in the initial individuals;
        - By default, this value is set to 1

    [-dm maximum_number_of_dimension]
        - This flag expects an integer with the maximum number of dimensions in each individual;
        - By default, this value is set to 9999

    [-rs random state]
        - This flag expects an integer with the seed to be used by the M6GP algorithm;
        - By default, this value is set to 42


	

How to import this implementation to your project:
    - Download this repository;
    - Copy the "m6gp/" directory to your project directory;
    - import the M6GP class using "from m6gp.M6GP import M6GP".

How to use this implementation:
    $ from m6gp.M6GP import M6GP
    $ model = M6GP()
    $ model.fit( training_x, training_y, test_x (optional), test_y (optional) )

Arguments for M6GP():
    operators		-> Operators used by the individual (default: [("+",2),("-",2),("*",2),("/",2)] )
    max_depth		-> Max initial depths of the individuals (default: 6)
    population_size	-> Population size (default: 500)
    max_generation	-> Maximum number of generations (default: 100)
    tournament_size	-> Tournament size (default: 5)
    limit_depth		-> Maximum individual depth (default: 17)
    threads 		-> Number of CPU threads to be used (default: 1)
    random_state	-> Random state (default: 42)
    dim_min		-> Minimum number of dimensions (default: 1)
    dim_max		-> Maximum number of dimensions (default: 9999) #The algorithm will not reach this value

Arguments for model.fit():
    Tr_X 		-> Training samples
    Tr_Y 		-> Training labels
    Te_X 		-> Test samples, used in the standalone version (default: None)
    Te_Y 		-> Test labels, used in the standalone version (default: None)

Useful methods:
    $ model = M6GP()	-> starts the model;
    $ model.fit(X, Y)	-> fits the model to the dataset;
    $ model.predict(X)	-> Returns a list with the prediction of the given dataset.




How to edit this implementation:
    Fitness Function ( m6gp.Individual ):
        - Change the getFitness() method to use your own fitness function;
        - This implementation assumes that a higher fitness is always better. To change this, edit the __gt__ method in this class;
        - Warning: Since M6GP is a slow method, a fitness function that escalates well with the number of features is recommended. 

    Classification/regression algorithm ( m6gp.Individual ):
        - Change the createModel() method to use your own classifier;
        - Assuming it is a scykit-learn implementation, you may only need to change one line in this method;
        - Warning: Since M6GP is a slow method, a learning algorithm that escalates well with the number of features is recommended.

   


Reference: 

@inproceedings{m6gp,
	doi = {10.1109/CEC60901.2024.10612107},
  	url = {https://ieeexplore.ieee.org/abstract/document/10612107},
  	year = {2024},
  	month = jun,
  	publisher = {{IEEE}},
  	author = {Joao E. Batista and Nuno M. Rodrigues and Leonardo Vanneschi},
  	title = {{M6GP: Multiobjective Feature Engineering}},
	booktitle = {2024 {IEEE} Congress on Evolutionary Computation ({CEC})}
}


You may also be interested in other works related to measuring the complexity of feature engineering models:
 - https://ieeexplore.ieee.org/abstract/document/10611989
 - https://www.sciencedirect.com/science/article/pii/S2210650224002992



