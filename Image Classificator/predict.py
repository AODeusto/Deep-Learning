import argparse

import model as m
import process as p

parser = argparse.ArgumentParser(description = 'Predict the name of a flower from a picture')

parser.add_argument('path_to_image', type = str, help = 'Path to the image file')
parser.add_argument('checkpoint', type = str, help = 'Path to the checkpoint to rebuild model from')
parser.add_argument('-t','--top_k', dest = 'top_k', type = int, default = 1, metavar = '', \
                    help = 'Select number of most likely classes')
parser.add_argument('-cat','--category_names ', dest = 'category_names', type = str, metavar = '', \
                    help = 'Dictionary mapping the integer encoded classes to real flower names')
parser.add_argument('-g','--gpu', dest = 'gpu', action = 'store_true', help = 'Turn ON GPU if available')

args = parser.parse_args()

path_to_image = args.path_to_image
checkpoint = args.checkpoint
top_k = args.top_k
category_names = args.category_names
gpu = args.gpu


m.predict_image_label(path_to_image, top_k, checkpoint, category_names, gpu)

#print(model.state_dict())
