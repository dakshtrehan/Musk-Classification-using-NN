# Musk-Classification-using-NN
This project uses Neural Network to classify Musks.

First we are importing the given dataset and dropping columns of ID, Molecule_name, Confronation_name because what all we need is features we can even convert dropped columns to number(vectorization).

Next we convert the dataframe to array and assing X,Y. X being input and Y being output.

Split the data into train and test and create a model with 5 layer(3 being hidden).

Compile the model using BinaryCrossEntropy and metrics as accuracy.

Fit the model with batch size in multiple of 2**n.

Using hist object plot the graph between validation loss, training loss and validation accuracy, training accuracy.

Check for the epochs(iterations) if validation loss is increasing and validation accuracy is decreasing perfrom early stopping by decreasing epochs.

Check for the accuracy on testing and training dataset.

Choices made :
1. Sigmoid is used as it will classify output as 0 or 1 but ReLu/SoftMax will provide probablities.
2. Epochs are chosen by focussing on decrease in validation accuracy and increase in validation loss for multiple times.
3. 5 layers were made because it was mentioned in the task.
4. ID, Molecule_name, Confrontation_name are dropped because we were concerned with features although those can be vectorized thus not causing any problem to model.

