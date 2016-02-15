# Homework Report

1)	Write a function that evaluates the trained network (5 points), as well as computes all the subgradients of W_1W1 and W_2W2 using backpropagation (5 points).

See code;

2)	Write a function that performs stochastic mini-batch gradient descent training (5 points). You may use the deterministic approach of permuting the sequence of the data. Use the momentum approach described in the course slides.

See code;

3)	Train the network on the attached 2-class dataset extracted from CIFAR-10: (data can be found in announcement on canvas.). The data has 10,000 training examples in 3072 dimensions and 2,000 testing examples. For this assignment, just treat each dimension as uncorrelated to each other. Train on all the training examples, tune your parameters (number of hidden units, learning rate, mini-batch size, momentum) until you reach a good performance on the testing set. What accuracy can you achieve? (20 points based on the report).

ANS: The best accuracy would be, on average, steady at 83%.
Tuning was applied in four nested loop in the main function.

4)	Code Description: please provide detailed code description for each part of your project in the report. Every function in your code must be explained with its functionality and usage (Maximal 2 points will be deducted from each part if this is not provided).

See code;

5)	Training Monitoring: For each epoch in training, your function should evaluate the training objective, testing objective, training misclassification error rate (error is 1 for each example if misclassifies, 0 if correct), testing misclassification error rate (5 points).

See terminal printed information when running the code.

(6) Tuning Parameters: please create three figures with following requirements. Save them into jpg format:
i) Test accuracy with different number of batch size
ii) Test accuracy with different learning rate
iii) Test accuracy with different number of hidden units

See JPG file in the folder.

(7) Discussion about the performance of your neural network.

ANS: The performance is subject to the value of parameters. In the figure, the performance increases as number of neuron increases. Short step size would converge slower than large step size, but will remain at a higher performance as epoch increases. And, as the figure shows, the accuracy increase faster when we choose small batch size of data for training.
From the figure of “tune step size”, we can see that the change of step size has the most significant impact on the performance. 
