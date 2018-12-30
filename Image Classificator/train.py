import argparse

import model as m
import process as p

parser = argparse.ArgumentParser(description = 'Train a model from a labeled images folder + create a checkpoint')

parser.add_argument('data_dir', type = str, help = 'Directory containing training, validation and testing folders of images')
parser.add_argument('-s','--save_dir ', dest = 'save_dir', type = str, metavar = '', help = 'Where to save checkpoint once NN is trained. Default data_dir')
parser.add_argument('-a','--arch', dest = 'arch', type = str, default = 'vgg', metavar = '', help = 'Select CNN Architecture. Choose between Alexnet, Vgg, or Densenet. Default VGG')
parser.add_argument('-lr','--learning_rate', dest = 'learning_rate', type = float, default = 0.01, metavar = '', help = 'Select Learning Rate for optimizer')
parser.add_argument('-hu','--hidden_units ', dest = 'hidden_units', type = int, default = 512, metavar = '', help = 'Number of units in the hidden layer')
parser.add_argument('-e','--epochs ', dest = 'epochs', type = int, metavar = '', default = 5, help = 'Number of epochs')
parser.add_argument('-g','--gpu', dest = 'gpu', action = 'store_true', help = 'Turn ON GPU if available')

args = parser.parse_args()

data_dir = args.data_dir
save_dir = args.data_dir if args.save_dir == None else args.save_dir
arch = args.arch
learning_rate = args.learning_rate
hidden_units = args.hidden_units
epochs = args.epochs
gpu = args.gpu

# model = m.define_pretrained_model(arch)
# model, criterion, optimizer, inputs, outputs, hidden_units = m.define_classificator_optim_error(learning_rate, hidden_units, data_dir, arch)
m.train_model(gpu, learning_rate, hidden_units, data_dir, arch, epochs, save_dir)

print(model)
print(criterion)
print(optimizer)
print(inputs)
print(outputs)
print(hidden_units)
