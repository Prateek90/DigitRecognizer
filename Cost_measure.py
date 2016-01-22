import mnist_loader
training_data, validation_data, test_data=mnist_loader.load_data_wrapper()
import newNetwork2
net=newNetwork2.Network([784,30,10],cost=newNetwork2.CrossEntropyCost)
net.SGD(training_data,30,0.001,evaluation_data=test_data,monitor_training_accuracy=True,monitor_evaluation_cost=True)
