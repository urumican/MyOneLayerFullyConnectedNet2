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

	# parameters 
	stepSizes = [0.001, 0.003, 0.005, 0.007, 0.009, 0.011, 0.013, 0.015, 0.017, 0.019]
	gammas = [0.6, 0.7, 0.8, 0.9]

	bestStepSize = 0
	bestBatchSize = 0
	bestNumOfNeurons = 0
	bestGamma = 0
	curAcc = 0
	globalOptimalAcc = 0
	res = {}

	# Tuning through stepSize, number of neuron, batch size;
	for stepSizesIdx in range(10):
		for numOfNeuron in range(10 , 101, 1):
			for batchSize in range(32, 258, 2):
				for gammaIdx in range(4):
					print "Gamma:", gammas[gammaIdx]
					# Create layer-level parameter
					# Parameter format:
					# [num, activation function, derivative of activation function]
					specification = ((train_data.shape[1], 0, 0), (numOfNeuron, relu, relu_prime), (1, sigmoid, sigmoid_prime))	

					# Start calculation
					curAcc = net.mySDGwithMomentum(param = specification, loss = crossEntropy, lossPrime = corssEngtropyPrime, miniBatchSize = batchSize, stepSize = stepSizes[stepSizesIdx], gamma = gammas[gammaIdx])
					if curAcc > globalOptimalAcc:
						bestStepSize = stepSizes[stepSizesIdx]
						bestBatchSize = batchSize
						bestNumOfNeurons = numOfNeuron
						bestGamma = gammas[gammaIdx]
						globalOptimalAcc = curAcc
					#end

					print "\n \n \n", "Current Acc: ", curAcc, "\n \n \n"

					res["bestStepSize"] = stepSizes[stepSizesIdx]
					res["bestBatchSize"] = batchSize
					res["bestNumOfNeurons"] = numOfNeuron
					res["bestGamma"] = gammas[gammaIdx]
					res["curAcc"] = curAcc
					cPickle.dump(res, open("/nfs/stak/students/l/lix3/dl/try/myRes/global/" + "_" + str(stepSizesIdx) + "_" + str(batchSize) + "_" + str(numOfNeuron) + "_" + str(gammaIdx) +".p", "wb"))
				# end
			# end
		# end
	# end
	
	print "Best step size: ", bestStepSize 
	print "Best batch size: ", bestBatchSize
	print "Best number of neurons: ", bestNumOfNeurons
	print "Best gamma(momentum): ", bestGamma
	print "Best Acc:", globalOptimalAcc

# end #

if __name__ == "__main__":
    main()
