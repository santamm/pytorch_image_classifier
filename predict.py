import matplotlib
import matplotlib.pyplot as plt
import torch
import numpy as np
import argparse
#matplotlib.use('tkagg')
#from torch import nn
#from torch import optim
#import torch.nn.functional as F
#from torchvision import datasets, transforms, models
import load
from collections import OrderedDict
from PIL import Image
import json
import seaborn as sns



def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # copy image to avoid modifying original one
    img1 = image.copy()
    # Resize the image where the shortest side is 256 pixels
    if img1.width > img1.height:
        img1.thumbnail((img1.width, 256))
    else:
        img1.thumbnail((256, img1.height))

    # Crop the image to 224x224
    img1 = img1.crop(((img1.width-224)/2, (img1.height-224)/2, (img1.width-224)/2 + 224, (img1.height-224)/2 + 224))

    # Normalizing
    np_image = np.array(img1)
    np_image = np_image / 255.
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    np_image = (np_image - means) / stds

    # Reordering dimensions
    np_image = np.transpose(np_image, axes = (2, 0, 1))

    return np_image


def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)
    return ax

def load_class_to_idx(loadpath):
    # Load class to idx dictionary
    checkpoint = torch.load(loadpath, map_location=lambda storage, loc: storage)
    return checkpoint['class_to_idx']

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = process_image(Image.open(image_path))
    if device.type=='cuda':
        t_img = torch.from_numpy(image).float().cuda()
    else:
        t_img = torch.from_numpy(image).float()
    t2 = t_img.resize(1, 3, 224, 224)
    with torch.no_grad():
        outputs=model.forward(t2)
        (probs, indices) = torch.topk(torch.exp(outputs), topk)
        if device.type=='cuda':
            n_indices = indices.cpu().numpy()
            n_probs = probs.cpu().numpy()[0]    
        else:
            n_indices = indices.numpy()
            n_probs = probs.numpy()[0]
    classes = []
    for i in range (topk):
        classes.append(idx_to_class[n_indices[0][i]])
    return classes, n_probs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
    description='Example with nonoptional arguments',
    )

    parser.add_argument('input', action="store")
    parser.add_argument('checkpoint', action="store")
    parser.add_argument('--topk', action="store", dest="topk", type=int, default=3)
    parser.add_argument('--category_names ', action="store",
        dest="category_names", default="cat_to_name.json")
    parser.add_argument('--gpu', action="store_true", dest="gpu", default=False)

    results = parser.parse_args()
    #print('input     = {!r}'.format(results.input))
    #print('checkpoint   = {!r}'.format(results.checkpoint))
    #print('topk        = {!r}'.format(results.topk))
    #print('category names        = {!r}'.format(results.category_names))
    print('gpu        = {!r}'.format(results.gpu))


    model = load.load_model(results.checkpoint)
    if results.gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
    #criterion = load.load_criterion(results.checkpoint)
    #loaded_optimizer = load_optimizer('checkpoint.pth', loaded_model)

    class_to_idx = load_class_to_idx(results.checkpoint)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    with open(results.category_names, 'r') as f:
        cat_to_name = json.load(f)

    image_path = results.input

    (classes, probs) = predict(image_path, model, results.topk)
    flowers = []
    for i in range(results.topk):
        flowers.append(cat_to_name[classes[i]])

    image = Image.open(image_path)
    print("Predictions:")
    for cl in range(results.topk):
        print("Flower {} with probability {:.2f}%".format(flowers[cl], probs[cl]*100))
    plt.figure(figsize = (7,5))
    ax = plt.subplot(2, 1, 1)

 
    ax.set_title(flowers[0])
    ax.axis('off')
    img = process_image(image)
    imshow(img, ax, title = flowers[0]);

    # Plot bar chart

    plt.subplot(2,1,2)
    plt.xlabel('Probabilities (%)')
    plt.title('Top 5 Predictions')
    sns.barplot(x=probs*100, y=flowers, color=sns.color_palette()[0])
    plt.show()
