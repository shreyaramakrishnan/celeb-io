from uwimg import *

#  The layers should be inputs -> 64, 64 -> 32, and 32 -> outputs

def softmax_model(inputs, outputs):
    l = [make_layer(inputs, outputs, SOFTMAX)]
    return make_model(l)

def neural_net(inputs, outputs):
    print(inputs)
    l = [   make_layer(inputs, 64, RELU),
            make_layer(64, 32, RELU),
            make_layer(32, outputs, SOFTMAX)]
    return make_model(l)

print("loading data...")
train = load_classification_data(b"cifar.train", b"cifar/labels.txt", 1)
test  = load_classification_data(b"cifar.test",  b"cifar/labels.txt", 1)
print("done")
print

print("training model...")
batch = 128
iters = 3000
rate = pow(10, -1)
momentum = .9
decay = pow(10, -4)
# decay = 0


# m = softmax_model(train.X.cols, train.y.cols)
m = neural_net(train.X.cols, train.y.cols)
train_model(m, train, batch, iters, rate, momentum, decay)
print("done")
# print

print("evaluating model...")
print("training accuracy: %f", accuracy_model(m, train))
print("test accuracy:     %f", accuracy_model(m, test))

""" 
5.2.2.1 Question
Why might we be interested in both training accuracy and testing accuracy? What do these two numbers tell us about our current model?

Training accuracy that approaches 1 too quickly is likely overfitting and will lead to a lower testing accuracy so we 
want to reach a good balance between high testing and training accuracy. 
Training accuracy is the model's ability to fit the training set
while testing accuracy is the model's ability to fit a dataset it was not trained on.

5.2.2.2 Question
Try varying the model parameter for learning rate to different powers of 10 (i.e. 10^1, 10^0, 10^-1, 10^-2, 10^-3) and training the model. 
What patterns do you see and how does the choice of learning rate affect both the loss during training and the final model accuracy?

Rate = 10^0
training accuracy: %f 0.2068
test accuracy:     %f 0.2084
000999: Loss: 1.996511

Rate = 10^1
training accuracy: %f 0.09871666666666666
test accuracy:     %f 0.098

rate = 10^0
training accuracy: %f 0.8857833333333334
test accuracy:     %f 0.8917
000999: Loss: 0.500659

rate = .01 (10^-1)
training accuracy: %f 0.9204
test accuracy:     %f 0.9198
000999: Loss: 0.175899

rate = 10^-2
training accuracy: %f 0.90365
test accuracy:     %f 0.9111
000999: Loss: 0.251800

rate = 10^-3
training accuracy: %f 0.85815
test accuracy:     %f 0.8694
000999: Loss: 0.575920

The best learning rate was 10^-2, with rates above 1 having a very very low accuracy, and rates smaller than 10^-2
going down in a somewhat quadratic/exponential fashion as the exponent increased in magnitude.
A "better" learning rate tends to minimize loss and lead to better training and final model accuracies.

5.2.2.3 Question
Try varying the parameter for weight decay to different powers of 10: (10^0, 10^-1, 10^-2, 10^-3, 10^-4, 10^-5). How does weight decay affect the final model training and test accuracy?

All trials w/ rate of 10^-2
Decay = 10^0
training accuracy: %f 0.8986166666666666
test accuracy:     %f 0.906
000999: Loss: 0.325935

Decay = 10^-1
training accuracy: %f 0.90345
test accuracy:     %f 0.9109
000999: Loss: 0.259186

Decay = 10^-2
training accuracy: %f 0.9035833333333333
test accuracy:     %f 0.9111
000999: Loss: 0.252532

Decay = 10^-3
training accuracy: %f 0.9036333333333333
test accuracy:     %f 0.9111
000999: Loss: 0.251873

Decay = 10^-4
training accuracy: %f 0.90365
test accuracy:     %f 0.9111
000999: Loss: 0.251807

Decay = 10^-5
training accuracy: %f 0.90365
test accuracy:     %f 0.9111
000999: Loss: 0.251800

A lower decay value leads to lower loss and a higher accuracy on both the training and test datasets. The benefits
do appear to "converge" towards an optimal accuracy for a given rate. A smaller decay affects the weights less and allows for more
more precise fitting of the model but still with some "overfitting protection". There isn't a signicicant difference in the final result which
might be due to the fact that the model was not overfitting in the first place.

5.2.3 Train a neural network
Now change the training code to use the neural network model instead of the softmax model.

5.2.3.1 Question
Currently the model uses a logistic activation for the first layer. Try using a the different activation functions we programmed. How well do they perform? What's best?

Logistic:
training accuracy: %f 0.8866333333333334
test accuracy:     %f 0.8938
000999: Loss: 0.393381

RELU:
training accuracy: %f 0.9234333333333333
test accuracy:     %f 0.926
000999: Loss: 0.241253

LRELU:
training accuracy: %f 0.9214166666666667
test accuracy:     %f 0.9239
000999: Loss: 0.245111

The best one is RELU.

5.2.3.2 Question
Using the same activation, find the best (power of 10) learning rate for your model. What is the training accuracy and testing accuracy?

Rate = 10^0
training accuracy: %f 0.20588333333333333
test accuracy:     %f 0.2074
000999: Loss: 2.050483

Rate = 10^-1
training accuracy: %f 0.9617333333333333
test accuracy:     %f 0.9554
000999: Loss: 0.147100

Rate = 10^-2
training accuracy: %f 0.9235166666666667
test accuracy:     %f 0.9259
000999: Loss: 0.241040

Rate = 10^-3
training accuracy: %f 0.8633
test accuracy:     %f 0.8683
000999: Loss: 0.499928

**Best learning rate: 10^-1, training accuracy: 0.9617333333333333, test accuracy: 0.9554**

5.2.3.3 Question
Right now the regularization parameter decay is set to 0. Try adding some decay to your model. What happens, does it help? Why or why not may this be?

Decay = 10^-4
Rate = 10^0
training accuracy: %f 0.2068
test accuracy:     %f 0.2084
000999: Loss: 1.996511

Rate = 10^-1
training accuracy: %f 0.96118
test accuracy:     %f 0.9534

Rate = 10^-2
training accuracy: %f 0.9234333333333333
test accuracy:     %f 0.926
000999: Loss: 0.241253

Rate = 10^-3
training accuracy: %f 0.8633
test accuracy:     %f 0.8683
000999: Loss: 0.499931

Decay = 10^-1
Rate = 10^-1
Rate = 10^-2
training accuracy: %f 0.9226666666666666
test accuracy:     %f 0.9253
Rate = 10^-3
training accuracy: %f 0.86305
test accuracy:     %f 0.8678

The decay did not make the results better between a range of values. Although the purpose of decay
is to reduce the probablity of overfitting by reducing large weights, this might not be a benefit 
if the model was not yet overfitting (the learning rate is properly adjusted and the number of iterations properly matched).

5.2.3.4 Question
Modify your model so it has 3 layers instead of two. The layers should be inputs -> 64, 64 -> 32, and 32 -> outputs. Also modify your model to train for 3000 iterations instead of 1000. Look at the training and testing accuracy for different values of decay (powers of 10, 10^-4 -> 10^0). Which is best? Why?
Decay = 10^0:
training accuracy: %f 0.94635
test accuracy:     %f 0.95
Decay = 10^-1:
training accuracy: %f 0.97345
test accuracy:     %f 0.9627
Decay = 10^-2:
training accuracy: %f 0.9829666666666667
test accuracy:     %f 0.9713
Decay = 10^-3:
training accuracy: %f 0.9835
test accuracy:     %f 0.9694
Decay = 10^-4:
training accuracy: %f 0.9824666666666667
test accuracy:     %f 0.9678

The decay value of 10^-4 was the best. This makes sense because large magnitudes (10^-3 and up) simplify the model more and reduce larger weights more.
A smaller decay value allows for more fitting to the model which will increase accuracy while still avoiding extreme overfitting that would cause a decrease in accuracy.

5.3 Training on CIFAR
The CIFAR-10 dataset is meant to be similar to MNIST but much more challenging. Using the same model we just created, let's try training on CIFAR.

5.3.1 Get the data
We have to do a similar process as last time, getting the data and creating files to hold the paths. Run:

wget http://pjreddie.com/media/files/cifar.tgz
tar xzf cifar.tgz
find cifar/train -name \*.png > cifar.train
find cifar/test -name \*.png > cifar.test
Notice that the file of possible labels can be found in cifar/labels.txt

5.3.2 Train on CIFAR
Modify tryhw5.py to use CIFAR. This should mostly involve swapping out references to mnist with cifar during dataset loading. Then try training the network. You may have to fiddle with the learning rate to get it to train well.

5.3.2.1 Question
How well does your network perform on the CIFAR dataset? 
It performs "okay", with less accuracy than the MNIST but still at almost 50% accuracy, far better than random choice.
10^-2
training accuracy: %f 0.4633
test accuracy:     %f 0.4482
"""