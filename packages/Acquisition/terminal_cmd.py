import argparse


def terminal_parser():
	parser = argparse.ArgumentParser(description="Determines whether a cell has been infected by Malaria " \
												 "by looking a 2D picture of it.")
	parser.add_argument("-t", "--train", action="store_true",
						help="If summoned, loads data from data folder, trains the model and saves it as a h5 file.")
	parser.add_argument("-p", "--predict", action="store_true",
						help="If summoned, loads previously trained model and analyses whatever image passed using the -i flag")
	parser.add_argument("-i", "--image", type=str, metavar="", default="data/test/Parasitized-2.png",
						help="Cell image to analyse. Example tests images can be found at data/test")
	parser.add_argument("-r", "--report", action="store_true",
						help="If summoned, Saves the prediction as a pdf file at data/predictions. This flag must be used" 
						" alongside -pr or -pri")

	return parser.parse_args()


def str2bool(val):
	if isinstance(val, bool):
		return val
	if val.lower() in ('yes', 'true', 't', 'y', '1', 'si', 's'):
		return True
	elif val.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')
