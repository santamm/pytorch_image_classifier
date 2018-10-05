# Load model from saved checkpoint
import torch
from torch import nn
from torchvision import datasets, transforms, models

def load_model(filepath):
    # Force location otherwise it won't reaload on a CPU if saved on a GPU
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)

    # Reloading the pretrained model and freezing the weights
    pre_trained_model = getattr(models, checkpoint['pre_trained'])
    model = pre_trained_model(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    # Rebuilding the classifier
    classifier_input_size = checkpoint['classifier_input_size']
    classifier_hidden_layer = checkpoint['classifier_hidden_layer']
    output_size = checkpoint['output_size']
    model.classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(classifier_input_size, classifier_hidden_layer)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(classifier_hidden_layer, output_size)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    # Loading the saved weights
    model.load_state_dict(checkpoint['state_dict'])

    return model

def load_criterion(loadpath):

    # Rebuilding Loss Function
    checkpoint = torch.load(loadpath, map_location=lambda storage, loc: storage)
    loss_name = checkpoint['criterion']
    criterion = getattr(nn.modules.loss, loss_name)
    criterion = criterion()

    return criterion

def load_optimizer(loadpath, model):
    #Rebuilding Optimizer
    checkpoint = torch.load(loadpath, map_location=lambda storage, loc: storage)
    optimizer=getattr(torch.optim, checkpoint['optimizer_name'])
    learning_rate = checkpoint['learning_rate']
    optimizer = optimizer(model.classifier.parameters(), lr=learning_rate)
    state_dict=checkpoint['optimizer_state']
    optimizer.load_state_dict(state_dict())

    return optimizer


from collections import OrderedDict
