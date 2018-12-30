from torchvision import datasets, transforms, models
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import os
import re
import json

import process as pr

def define_pretrained_model(model_user):
    if 'alex' in model_user:
        model = models.alexnet(pretrained=True)
    elif 'vgg' in model_user:
        model = models.vgg16(pretrained=True)
    elif 'dense' in model_user:
        model = models.densenet161(pretrained=True)
#    elif 'squeezenet' in model_user:
#        model = models.squeezenet1_0(pretrained=True)
#    elif 'inception' in model_user:
#        model = models.inception_v3(pretrained=True)
#    if 'resnet' in model_user:
#        model = models.resnet18(pretrained=True)
    return model

def define_classificator_optim_error(learning_rate, hidden_units, data_dir, arch):
    model = define_pretrained_model(arch)
    inputs = int(re.findall(r'\d+', str(model.classifier)[str(model.classifier).find('Linear'): ])[0])
    outputs = len(os.listdir(os.path.join(data_dir, 'train')))
    my_classifier = nn.Sequential(nn.Linear(inputs, hidden_units),
                              nn.ReLU(),
                              nn.Dropout(0.2),
                              nn.Linear(hidden_units, int(hidden_units/2)),
                              nn.ReLU(),
                              nn.Dropout(0.2),
                              nn.Linear(int(hidden_units/2), outputs),
                              nn.LogSoftmax(dim = 1) )
    for param in model.parameters():
        param.requires_grad = False
    # Replace classifier in VGG model for the one defined above
    model.classifier = my_classifier
    # Define GPU, error function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
    return model, criterion, optimizer, inputs, outputs, hidden_units

def gpu_usage(gpu):
    if gpu == True:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print ('GPU ----> ON and ready to train')
        else:
            device = torch.device('cpu')
            print ('GPU ----> NOT AVAILABLE, ready to train with CPU')
    else:
        device = torch.device('cpu')
        print('Ready to train with CPU')
    return device

def train_model(gpu, learning_rate, hidden_units, data_dir, arch, epochs, save_dir):              ##############################################################
    device = gpu_usage(gpu)
    model, criterion, optimizer, inputs, outputs, hidden_units = \
            define_classificator_optim_error(learning_rate, hidden_units, data_dir, arch)
    dataloader, class_to_idx = pr.process_image_train(data_dir)
    # Send model to GPU (if available)
    model.to(device)
    print('Model sent to device')
    steps = 0
    print_every = 100
    # create lists to store train and validation losses
    train_losses= []
    validation_losses = []
    validation_accuracies = []
    current_train_loss = 0
    print ('........Training........')
    # for e in range(epochs):
    #     for images, labels in dataloader['train']:
    #         images, labels = images.to(device), labels.to(device)
    #         optimizer.zero_grad()
    #         # Forward pass
    #         log_ps = model.forward(images)
    #         loss = criterion(log_ps, labels)
    #
    #         # Backpropagate the error
    #         loss.backward()
    #         optimizer.step()
    #
    #         # Update steps and training error
    #         steps += 1
    #         current_train_loss += loss.item()
    #
    #         # Validation each 5 steps
    #         if steps % print_every == 0:
    #             current_valid_loss = 0
    #             accuracy = 0
    #             # Turn-off Dropout, we want the whole classifier to be working
    #             model.eval()
    #             # Turn-off gradients
    #             with torch.no_grad():
    #                 for images, labels in dataloader['valid']:
    #                     images, labels = images.to(device), labels.to(device)
    #
    #                     log_ps = model.forward(images)
    #                     loss = criterion(log_ps, labels)
    #
    #                     ps = torch.exp(log_ps)
    #                     top_ps, top_class = ps.topk(1, dim=1)
    #                     equal = (top_class == labels.view(*top_class.shape))
    #                     batch_accuracy = torch.mean(equal.type(torch.FloatTensor))
    #
    #                     current_valid_loss += loss.item()
    #                     accuracy += batch_accuracy.item()
    #             model.train()
    #
    #             train_losses.append(current_train_loss / print_every)
    #             validation_losses.append(current_valid_loss / len(dataloader['valid']))
    #             validation_accuracies.append((accuracy * 100) / len(dataloader['valid']))
    #
    #             print ('epoch: {}/{}'.format((e+1), epochs),
    #                    'step: {}'.format(steps),
    #                    'train err: {:.3f}'.format(current_train_loss / print_every),
    #                    'valid err: {:.3f}'.format(current_valid_loss / len(dataloader['valid'])),
    #                    'acc: {:3f}%'.format( (accuracy * 100) / len(dataloader['valid']) ))
    #
    #             current_train_loss = 0

    model.class_to_idx = class_to_idx
    checkpoint = {'input_size' : inputs,
                  'hidden' : hidden_units,
                  'output_size' : outputs,
                  'epoch' : epochs,
                  'lr' : float(str(optimizer)[str(optimizer).find('lr') : ].split()[1]),
                  'model_name' : type(model),
                  'state_dict' : model.state_dict(),
                  'class_to_idx' : model.class_to_idx,
                  'optim_state_dict' : optimizer.state_dict() }

    torch.save(checkpoint, os.path.join(data_dir, 'checkpoint.pth')) ###### QUIZAS HAGA FALTA CAMBIAR path A checkpoint SOLO  #####

def load_checkpoint(checkpoint_path, gpu):
    if gpu == True:
        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_path)
            device = torch.device('cuda')
            print('Using GPU for prediction')
        else:
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
            device = torch.device('cpu')
            print('Using CPU for prediction, because CUDA is not available')
    else:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        device = torch.device('cpu')
        print('Using CPU for prediction')
    print ('Loading checkpoint')
    model_name_find = str(checkpoint['model_name'])[str(checkpoint['model_name']).rfind('.')+1 : ].lower()
    model = define_pretrained_model(model_name_find)
    my_classifier = nn.Sequential(nn.Linear(checkpoint['input_size'], checkpoint['hidden']),
                              nn.ReLU(),
                              nn.Dropout(0.2),
                              nn.Linear(checkpoint['hidden'], int(checkpoint['hidden'] / 2)),
                              nn.ReLU(),
                              nn.Dropout(0.2),
                              nn.Linear(int(checkpoint['hidden'] / 2), checkpoint['output_size']),
                              nn.LogSoftmax(dim = 1) )

    for param in model.parameters():
        param.requires_grad = False
    model.classifier = my_classifier
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = checkpoint['lr'])      ######### COMPROBAR QUE FUNCIONA #######
#    optimizer.load_state_dict(checkpoint['optim_state_dict'])
    print ('Model,Criterion and Optimizer loaded.')
    return model, criterion, optimizer, device

def predict_image_label(image_path, topk, checkpoint_path, cat_names_dict, gpu):
    model, criterion, optimizer, device = load_checkpoint(checkpoint_path, gpu)
    print('....Processing image')
    image = pr.process_image_predict(image_path)
    print('Image processed')
    # Convert array to FloatTensor
    image_torch = torch.from_numpy(image).type(torch.FloatTensor)
    # Add first dimension to match the expected shape from the model (same as with the dataloader)
    image_torch.unsqueeze_(0)
    with torch.no_grad():
        model.eval()
        log_ps = model.forward(image_torch)
        ps = torch.exp(log_ps)
        top_probs, top_classes = ps.topk(topk, dim=1)
        if topk == 1:  ################################################################
            top_probs, top_classes = top_probs.squeeze(), top_classes.squeeze()
        else:
            top_probs, top_classes = list(top_probs.numpy().squeeze()), list(top_classes.numpy().squeeze())
    model.train()

    with open(cat_names_dict, 'r') as f:
        names = json.load(f)
    top_names = []
    if topk == 1:
        for key,value in model.class_to_idx.items():
            if top_classes == value:
                top_names.append(names[key].title())
        print('{}. {:.3f} probability'.format(top_names[0], top_probs))

    else:
        for i in top_classes:
            for key,value in model.class_to_idx.items():
                if i == value:
                    top_names.append(names[key])
        Up_top_names = []
        for i in top_names:
            Up_top_names.append(i.title())
        for i in range(len(Up_top_names)):
            print('{}. {:.3f} probability'.format(Up_top_names[i], top_probs[i]))
