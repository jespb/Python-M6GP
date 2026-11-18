import argparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor

#
# By using this file, you are agreeing to this product's EULA
#
# This product can be obtained in https://github.com/jespb/Python-M6GP
#
# Copyright ©2023-2025 J. E. Batista
#


# Models that can be wrapped by M6GP (-model)
MODELS = {
	"rf":  lambda: RandomForestClassifier(max_depth=6),
	"dtr": lambda: DecisionTreeRegressor(max_depth=6),
}


def parse_operators(s):
	"""'+,2;-,2;log2,1' -> [('+',2), ('-',2), ('log2',1)].
	Only operators implemented in m6gp.Node.calculate are available."""
	try:
		return [(name, int(arity)) for name, arity in (op.split(",") for op in s.split(";"))]
	except ValueError:
		raise argparse.ArgumentTypeError("expected 'op,arity;op,arity;...', got %r" % s)


def semicolon_list(s):
	return s.split(";")


def get_args(argv=None):
	p = argparse.ArgumentParser(description="M6GP standalone runner",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	p.add_argument("-dsdir", default="datasets/", help="datasets directory")
	p.add_argument("-odir",  default="results/", help="output directory")
	p.add_argument("-d",     dest="datasets", type=semicolon_list, default=["heart.csv"],
		help="';'-separated list of datasets")
	p.add_argument("-runs",  type=int, default=30, help="number of runs (run r uses seed r)")
	p.add_argument("-op",    dest="operators", type=parse_operators,
		default=parse_operators("+,2;-,2;*,2;/,2"), help="operators, e.g. '+,2;-,2;log2,1'")
	p.add_argument("-md",    dest="max_depth", type=int, default=6, help="max initial depth")
	p.add_argument("-ld",    dest="limit_depth", type=int, default=17, help="max dimension depth")
	p.add_argument("-ps",    dest="population_size", type=int, default=500)
	p.add_argument("-mg",    dest="max_generation", type=int, default=100)
	p.add_argument("-tf",    dest="train_fraction", type=float, default=0.70)
	p.add_argument("-ts",    dest="tournament_size", type=int, default=5)
	p.add_argument("-t",     dest="threads", type=int, default=1)
	p.add_argument("-dmin",  type=int, default=1, help="min number of dimensions")
	p.add_argument("-dmax",  type=int, default=9999, help="max number of dimensions")
	p.add_argument("-model", choices=MODELS, default="rf", help="wrapped model")
	p.add_argument("-fit",   dest="fitnesses", type=semicolon_list, default=["2FOLD", "Size"],
		help="';'-separated fitnesses")
	p.add_argument("-overwrite", action="store_true", help="overwrite existing result files")
	p.add_argument("-s",     dest="verbose", action="store_false", help="silent mode")

	args = p.parse_args(argv)
	args.model_name = MODELS[args.model]().__class__.__name__
	args.model = MODELS[args.model]()
	return args
