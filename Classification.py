import numpy
import cPickle
#from scipy.special import expit

# import my scripts
from MyOneLayerFullyConnectedNet import *
from myFunctions import *

def main():
	# Import data
	dic = cPickle.load(open("cifar_2class_py2.p","rb"))
	train_data = (dic['train_data'] - dic['train_data'].mean(0)) / dic['train_data'].std(0) 
	train_data.shape = (10000, 3072)
	#train_data = dic['train_data'] / 255
	train_label = dic['train_labels']	
	test_data = (dic['test_data'] - dic['test_data'].mean(0)) / dic['test_data'].std(0)
	test_data.shape = (2000, 3072)
	#test_data = dic['test_data'] / 255
	test_label = dic['test_labels'] 

	# Create new net
	net = MyOneLayerFullyConnectedNet(train_data, train_label, test_data, test_label)


	#for stepSIze in range(0.001, 1, 0.002):
	
	# Create layer-level parameter
	# Parameter format:
	# [num, activation function, derivative of activation function]
	specification = ((train_data.shape[1], 0, 0), (10, relu, relu_prime), (1, sigmoid, sigmoid_prime))	

	# Start calculation
	net.mySDGwithMomentum(param = specification, loss = crossEntropy, lossPrime = corssEngtropyPrime)
	
# end #

if __name__ == "__main__":
    main()
