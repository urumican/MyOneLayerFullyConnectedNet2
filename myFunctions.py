import numpy

def relu(x):
	tmp = x
	#print tmp
	tmp[numpy.where(tmp <= 0)] = 0;
	return tmp
# end #

def relu_prime(x):
	tmp = x
	tmp[numpy.where(tmp <= 0)] = 0
	tmp[numpy.where(tmp > 0)] = 1
	return tmp
# end #


def crossEntropy(x, y):
	#print 'output:', x
	return  -(y * numpy.log(x) + (1.0 - y) * numpy.log(1.0 - x))
# end #

def corssEngtropyPrime(x, y):
	return (y - x) / (x * (1.0 - x))
# end # 

def sigmoid(x):
	return (1.0 / (1.0 + numpy.exp(-x)))
# end #

def sigmoid_prime(x):
	return numpy.exp(-x) / (1.0 + numpy.exp(-x))**2
