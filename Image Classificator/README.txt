
Deep Learning Image Classificator. This project contains:

    - A Jupyter Notebook with the development of a Neural Network
    using Pytorch and training it with a flowers images Dataset. It involves
    image preprocessing, training, validation and testing of the NN, before
    infering it to predict flower´s names.

    - A Command Line Application to train the NN with your own labeled Dataset,
    save a checkpoint of the trained model, and predict the flowers names with
    higher probability. The train.py file allows the user to define the
    architecture of the NN, learning rate, number of hidden units, number of
    epochs, gpu usage, and directory to save a checkpoint from the trained
    model. The predict.py file loads the given checkpoint and image to perform
    the prediction. It takes arguments allowing the user to specify usage of
    gpu, path to a dictionary mapping classes to real flower names, and number
    of most likely names to show as a result.
