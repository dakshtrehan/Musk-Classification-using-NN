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
