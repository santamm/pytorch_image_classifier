import matplotlib.pyplot as plt
import torch
import numpy as np
import argparse
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    

def do_deep_learning(model, trainloader, epochs, print_every, criterion, optimizer, device='cpu'):
    epochs = epochs
    print_every = print_every
    steps = 0

    # change to cuda
    model.to(device)
    model.train()
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    validation_loss, validation_accuracy = validation(model, validloader, criterion, device)
                    print("Epoch: {}/{}... ".format(e+1, epochs),
                     "Running Loss: {:.4f}".format(running_loss/print_every),
                     "Validation Loss: {:.4f}".format(validation_loss),
                     "Validation Accuracy: {:.2f} %%".format(validation_accuracy))
                running_loss = 0
                # Make sure training is back on
                model.train()



def validation(model, loader, criterion, device='cpu'):
    correct = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            outputs = torch.exp(outputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return total_loss/len(loader), (100 * correct / total)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
    description='Example with nonoptional arguments',
    )
    parser.add_argument('data_dir', action="store")
    parser.add_argument('--save_dir', action="store", dest="save_dir")
    parser.add_argument('--arch ', action="store", dest="arch", default='vgg13',
                       choices=('vgg11', 'vgg13', 'vgg16', 'vgg19'))
    parser.add_argument('--learning_rate', action="store", dest="learning_rate",
    type=float,  default=0.01)
    parser.add_argument('--hidden_units', action="store", dest="hidden_units",
    type=int,  default=512)
    parser.add_argument('--epochs', action="store", dest="epochs", type=int,  default=20)
    parser.add_argument('--gpu', action="store_true", dest="gpu", default=False)

    args = parser.parse_args()

    input_size = 3*224*224
    output_size = 102
    classifier_input_size = 25088
    classifier_hidden_layer = args.hidden_units

    # Import model
    pre_trained_model = getattr(models, args.arch)
    model = pre_trained_model(pretrained=True)
    # Freeze weights
    for param in model.parameters():
        param.requires_grad = False

    from collections import OrderedDict
    # replace classifier
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(classifier_input_size, classifier_hidden_layer)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(classifier_hidden_layer, output_size)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    model.classifier = classifier

    if args.gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Loss function
    criterion = nn.NLLLoss()

    # Optimizer
    learning_rate = args.learning_rate
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # Data Loaders
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

    epochs = args.epochs

    do_deep_learning(model, trainloader, epochs, 20, criterion, optimizer, device)

    # save the model
    model.class_to_idx = train_data.class_to_idx


    checkpoint = {'input_size': input_size,
                  'output_size': output_size,
                  'classifier_input_size': classifier_input_size,
                  'classifier_hidden_layer': classifier_hidden_layer,
                  'pre_trained' : args.arch,
                  'optimizer_name' : 'Adam',
                  'optimizer_state': optimizer.state_dict,
                  'criterion': 'NLLLoss',
                  'epochs': args.epochs,
                  'device': device.type,
                  'learning_rate': args.learning_rate,
                  'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict()}

    if args.save_dir:
        saved_checkpoint = args.save_dir+"/checkpoint.pth"
    else:
        saved_checkpoint = "checkpoint.pth"
    torch.save(checkpoint, saved_checkpoint)
    print("Checkpoint saved at {}".format(saved_checkpoint))
    