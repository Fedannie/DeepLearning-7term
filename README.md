# First assignment
## Task 1:
Implement different optimization methods wihout using autograd or torch.optim. You are only allowed to use pytorch as yur numeric computation framework. The only exception is visualize function.

## Task 2:
- Implement simple fully-convolutional neural architecture for classification. 
- Make sure it is small enought to run on your home machine.
- Provide dataset visulization.
- Provide train/test split and validation
#### Requirements:
- Architecture should derive from torch.nn.Module
- Use torch.utils.data.Dataset and torch.utils.data.DataLoader. But if you manage co simplify this step using dataset torchivision, I will only encourage you.
- Implement at least one data transformer, but make sure it is useful for classification task.
- Use FashionMNIST dataset https://github.com/zalandoresearch/fashion-mnist
- Make sure you can fix random seed for all components of your code to make experiments reproducible
- Since you architecure should be fully-convolutional, make sure it does not depend on input size.

# Dependencies:
- jupyter notebook
- numpy
- pytorch, torchvision
- tensorboardX
- matplotlib
