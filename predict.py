## executed command:
## python predict.py flowers/test/1/image_06743.jpg checkpoint --category_names cat_to_name.json --gpu

import argparse
import numpy as np
import pandas as pd
from collections import OrderedDict
import torch
from torch import nn
from torchvision import datasets, transforms, models

import json
from PIL import Image

# Code hints from documentation:
# https://docs.python.org/3/library/argparse.html
# https://docs.python.org/3/library/argparse.html#the-add-argument-method
# Code hints from "Intro to Python. Lesson 6: Classifying Images"


'''
Predict flower name from an image with predict.py along with the probability of that name. 
That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

Basic usage: python predict.py /path/to/image checkpoint

Options:
Return top KK most likely classes: python predict.py input checkpoint --top_k 3
Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
Use GPU for inference: python predict.py input checkpoint --gpu
'''

def main():

    def get_input_args():
        
        parser = argparse.ArgumentParser()

        parser.add_argument('input', action='store', help='The directory of the input image')
        parser.add_argument('checkpoint', action='store', help='The directory of the checkpoint file to load')
        parser.add_argument('--top_k', type=str, dest='top_k', default='3', help='Top K cases')
        parser.add_argument('--category_names', type=str, dest='cat_to_name', help='Path to the json file with the category names')
        parser.add_argument('--gpu', dest='gpu', action='store_true', help='Use GPU')

        return parser.parse_args()

    in_args = get_input_args()

    print(
        "Command Line Arguments:"
        "\n input = ", in_args.input,
        "\n checkpoint = ", in_args.checkpoint,
        "\n top_k = ", in_args.top_k,
        "\n category_names = ", in_args.cat_to_name,
        "\n gpu = ", in_args.gpu,
         )

    input = in_args.input
    top_k = in_args.top_k
    checkpoint = in_args.checkpoint
    cat_to_name = in_args.cat_to_name
    gpu = in_args.gpu
   
    # Hardware definitions #
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Model
    model = models.vgg16(pretrained=True)

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # Classifier architechture
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, 4096)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(4096, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    # Replace Classifier
    model.classifier = classifier
        
    # Load model
    def load_checkpoint(filepath):
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['state_dict'])
        
        arch = checkpoint['arch']
        classifier = checkpoint['classifier']
        model.class_to_idx = checkpoint['class_to_idx']
        model.classifier = checkpoint['classifier']

        for param in model.parameters():
            param.requires_grad = False

        return model
    
    model = load_checkpoint('checkpoint.pth')
    print("Your model has been loaded!")


    def process_image(image):

        # Resize
        image.resize((256, 256))       
        #Crop
        width, height = image.size   # Get dimensions
        new_width = 224
        new_height = 224
        left = (width - new_width)/2
        top = (height - new_height)/2
        right = (width + new_width)/2
        bottom = (height + new_height)/2
        image =image.crop((left, top, right, bottom))    
        # Convert to Array
        np_image = np.array(image)
        # Normalize
        means = np.array([0.485, 0.456, 0.406])
        stds = np.array([0.229, 0.224, 0.225])
        image = (np_image/255 - means) / stds    
        # Transpose
        image = image.transpose((2, 0, 1))

        # Return
        return image

    def predict(image_path, model, topk=5):
        # open image
        image = Image.open(image_path)
        # image -> tensor
        image = torch.from_numpy(process_image(image))
        # return tensor of size one
        image = image.unsqueeze(0)
        # Double model to match tensor
        model = model.double()
        # pass tensor through model
        model.to(device)
        output = torch.exp(model.forward(image.to(device)))
        # get 5 largest values and indices
        l_values = output.topk(topk)
        # make them arrays
        probs, indexes = l_values[0].data[0].tolist(), l_values[1].data[0].tolist()
        # convert indexes to classes
        idx_to_class = {idx: classification for classification, idx in model.class_to_idx.items()}
        classes = []
        for i in indexes:
            classes.append(idx_to_class[i])
        # Return
        return probs, classes

    def classify_image(probs, classes, cat_to_name): 
    # Code hints from Urko Pineda @GitLab
        print("Classifying image...")
        with open(cat_to_name, 'r') as f:
            data = json.load(f)
            name_classes = [data[i] for i in classes]
            result = pd.DataFrame({
                'class': pd.Series(data=name_classes),
                'probability': pd.Series(data=probs, dtype='float64')
            })
            print("These are your results: ")
            print(result)
            

    # Prediction
    probs, classes = predict(input, model, int(top_k))

    # Classification    
    classify_image(probs, classes, cat_to_name)

if __name__ == '__main__':
     main()

