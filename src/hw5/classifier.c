#include <math.h>
#include <stdlib.h>
#include "image.h"
#include "matrix.h"

// Run an activation function on each element in a matrix,
// modifies the matrix in place
// matrix m: Input to activation function
// ACTIVATION a: function to run
void activate_matrix(matrix m, ACTIVATION a)
{
    int i, j;
    for(i = 0; i < m.rows; ++i){
        double sum = 0;
        for(j = 0; j < m.cols; ++j){
            double x = m.data[i][j];
            if(a == LOGISTIC){
                // TODO
                m.data[i][j] = 1 / (1 + exp(-1 * x));
            } else if (a == RELU){
                // TODO
                if ( x > 0) {
                  m.data[i][j] = x;
                } else {
                  m.data[i][j] = 0;
                }
            } else if (a == LRELU){
                // TODO
                if (x > 0) {
                  m.data[i][j] = x;
                } else {
                  m.data[i][j] = 0.1 * x;
                }
            } else if (a == SOFTMAX){
                // TODO
                // Remember that for our softmax activation we will take e^x for every element x, 
                // but then we have to normalize each element by the sum as well. 
                // Each row in the matrix is a separate data point so we want to 
                // normalize over each data point separately.
                m.data[i][j] = exp(x);
            }
            sum += m.data[i][j];
        }
        if (a == SOFTMAX) {
            // TODO: have to normalize by sum if we are using SOFTMAX
            // already within rows
            for (int k = 0; k < m.cols; k++) {
              m.data[i][k] = m.data[i][k] / sum;
            }
        }
    }
}

// Calculates the gradient of an activation function and multiplies it into
// the delta for a layer
// matrix m: an activated layer output
// ACTIVATION a: activation function for a layer
// matrix d: delta before activation gradient
void gradient_matrix(matrix m, ACTIVATION a, matrix d)
{
    int i, j;
    for(i = 0; i < m.rows; ++i){
        for(j = 0; j < m.cols; ++j){
            double x = m.data[i][j];
            // TODO: multiply the correct element of d by the gradient
            // To do that we take each element in our delta 
            // (the partial derivative) and multiply it by the 
            // gradient of our activation function.
            // linear/softmax gradient: 1 everywhere
            double gradient = 1.0;
            if (a == LOGISTIC) {
              // f'(x) = f(x) * (1 - f(x))
              gradient = x * (1 - x);
            } else if (a == RELU) {
              if (x <= 0) {
                gradient = 0;
              }
            } else if (a == LRELU) {
              if (x <= 0) {
                gradient = 0.1;
              }
            }
            d.data[i][j] = d.data[i][j] * gradient;
        }
    }
}

// Forward propagate information through a layer
// layer *l: pointer to the layer
// matrix in: input to layer
// returns: matrix that is output of the layer
matrix forward_layer(layer *l, matrix in)
{

    l->in = in;  // Save the input for backpropagation


    // TODO: fix this! multiply input by weights and apply activation function.
    matrix out = matrix_mult_matrix(l->in, l->w);
    activate_matrix(out, l->activation);

    free_matrix(l->out);// free the old output
    l->out = out;       // Save the current output for gradient calculation
    return out;
}

// Backward propagate derivatives through a layer
// layer *l: pointer to the layer
// matrix delta: partial derivative of loss w.r.t. output of layer
// returns: matrix, partial derivative of loss w.r.t. input to layer
matrix backward_layer(layer *l, matrix delta)
{
    // 1.4.1
    // delta is dL/dy
    // TODO: modify it in place to be dL/d(xw)q
    // need to take delta and gradient by w and store to delta again
    // output, activation, delta layer 
    gradient_matrix(l->out, l->activation, delta);

    // 1.4.2
    // TODO: then calculate dL/dw and save it in l->dw
    free_matrix(l->dw);
    matrix xt = transpose_matrix(l->in);
    matrix dw = matrix_mult_matrix(xt, delta); // delta is dL/d(xw)
    l->dw = dw;
    free_matrix(xt);

    
    // 1.4.3
    // TODO: finally, calculate dL/dx and return it.
    matrix wt = transpose_matrix(l->w);
    matrix dx = matrix_mult_matrix(delta, wt); // replace this
    free_matrix(wt);

    return dx;
}

// Update the weights at layer l
// layer *l: pointer to the layer
// double rate: learning rate
// double momentum: amount of momentum to use
// double decay: value for weight decay
void update_layer(layer *l, double rate, double momentum, double decay)
{
    // TODO:
    // Calculate Δw_t = dL/dw_t - λw_t + mΔw_{t-1}
    // decay = lambda, momentum = m, rate = weird n
    matrix dw_t_1 = axpy_matrix(-1 * decay, l->w, l->dw);
    matrix dw_t = axpy_matrix(momentum, l->v, dw_t_1);
    free_matrix(dw_t_1);

    // save it to l->v
    free_matrix(l->v);
    l->v = dw_t;

    // Update l->w
    l->w = axpy_matrix(rate, l->v, l->w);

    // Remember to free any intermediate results to avoid memory leaks

}

// Make a new layer for our model
// int input: number of inputs to the layer
// int output: number of outputs from the layer
// ACTIVATION activation: the activation function to use
layer make_layer(int input, int output, ACTIVATION activation)
{
    layer l;
    l.in  = make_matrix(1,1);
    l.out = make_matrix(1,1);
    l.w   = random_matrix(input, output, sqrt(2./input));
    l.v   = make_matrix(input, output);
    l.dw  = make_matrix(input, output);
    l.activation = activation;
    return l;
}

// Run a model on input X
// model m: model to run
// matrix X: input to model
// returns: result matrix
matrix forward_model(model m, matrix X)
{
    int i;
    for(i = 0; i < m.n; ++i){
        X = forward_layer(m.layers + i, X);
    }
    return X;
}

// Run a model backward given gradient dL
// model m: model to run
// matrix dL: partial derivative of loss w.r.t. model output dL/dy
void backward_model(model m, matrix dL)
{
    matrix d = copy_matrix(dL);
    int i;
    for(i = m.n-1; i >= 0; --i){
        matrix prev = backward_layer(m.layers + i, d);
        free_matrix(d);
        d = prev;
    }
    free_matrix(d);
}

// Update the model weights
// model m: model to update
// double rate: learning rate
// double momentum: amount of momentum to use
// double decay: value for weight decay
void update_model(model m, double rate, double momentum, double decay)
{
    int i;
    for(i = 0; i < m.n; ++i){
        update_layer(m.layers + i, rate, momentum, decay);
    }
}

// Find the index of the maximum element in an array
// double *a: array
// int n: size of a, |a|
// returns: index of maximum element
int max_index(double *a, int n)
{
    if(n <= 0) return -1;
    int i;
    int max_i = 0;
    double max = a[0];
    for (i = 1; i < n; ++i) {
        if (a[i] > max){
            max = a[i];
            max_i = i;
        }
    }
    return max_i;
}

// Calculate the accuracy of a model on some data d
// model m: model to run
// data d: data to run on
// returns: accuracy, number correct / total
double accuracy_model(model m, data d)
{
    matrix p = forward_model(m, d.X);
    int i;
    int correct = 0;
    for(i = 0; i < d.y.rows; ++i){
        if(max_index(d.y.data[i], d.y.cols) == max_index(p.data[i], p.cols)) ++correct;
    }
    return (double)correct / d.y.rows;
}

// Calculate the cross-entropy loss for a set of predictions
// matrix y: the correct values
// matrix p: the predictions
// returns: average cross-entropy loss over data points, 1/n Σ(-ylog(p))
double cross_entropy_loss(matrix y, matrix p)
{
    int i, j;
    double sum = 0;
    for(i = 0; i < y.rows; ++i){
        for(j = 0; j < y.cols; ++j){
            sum += -y.data[i][j]*log(p.data[i][j]);
        }
    }
    return sum/y.rows;
}


// Train a model on a dataset using SGD
// model m: model to train
// data d: dataset to train on
// int batch: batch size for SGD
// int iters: number of iterations of SGD to run (i.e. how many batches)
// double rate: learning rate
// double momentum: momentum
// double decay: weight decay
void train_model(model m, data d, int batch, int iters, double rate, double momentum, double decay)
{
    int e;
    for(e = 0; e < iters; ++e){
        data b = random_batch(d, batch);
        matrix p = forward_model(m, b.X);
        fprintf(stderr, "%06d: Loss: %f\n", e, cross_entropy_loss(b.y, p));
        matrix dL = axpy_matrix(-1, p, b.y); // partial derivative of loss dL/dy
        backward_model(m, dL);
        update_model(m, rate/batch, momentum, decay);
        free_matrix(dL);
        free_data(b);
    }
}

// image/label get_prediction(model m, image im, )


/* Questions
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
*/



