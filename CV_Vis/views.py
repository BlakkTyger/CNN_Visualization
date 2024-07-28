from django.shortcuts import render
from django.http import HttpResponse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from django.views.decorators.csrf import csrf_exempt
import os
import numpy as np
import json
from django.http import JsonResponse

def ML(img_arr):
    if torch.cuda.is_available():
        device = 'cpu'
    else:
        device = 'cpu'

    # Define the model architecture (Convolutional Neural Network)
    class CNN(nn.Module):
      def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 3)  # Input channels, output channels, kernel size
        self.pool = nn.MaxPool2d(2, 2)  # Kernel size, stride
        self.conv2 = nn.Conv2d(4, 4, 3)
        self.fc1 = nn.Linear(4 * 5 * 5, 120)  # Input features, output features
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # Output layer for 10 digits

      def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) 
        # print(x.shape) # Apply ReLU activation
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.shape)
        x = x.view(-1, 4 * 5 * 5)  # Flatten for fully-connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    model = CNN()
    import os
    pat = os.path.dirname(os.path.realpath(__file__))
    model.load_state_dict(torch.load(os.path.join(pat,"mnist.pth")))
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensors
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize based on MNIST statistics
    ])

    # Load and preprocess the image
    input_tensor = transform(np.float32(np.array(img_arr)))
    input_tensor = input_tensor.unsqueeze(0)  # Add a batch dimension

    with torch.no_grad():  # Disable gradient computation
        output = model(input_tensor)
    
    return output

@csrf_exempt
def process_canvas_data(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        array_data = data['data']
        result = ML(array_data).tolist()
        request.session['result'] = [result, array_data]
        print(result)
        return JsonResponse({'status': 'success', 'result': result})
    return JsonResponse({'status': 'error'})

def input(request):
    return render(request, 'input.html')

def animation(request):
    result = request.session.get('result', None)
    print("lmao")
    print(result)
    print("lmao")
    context = {'result': result}
    return render(request, 'animation.html', context)