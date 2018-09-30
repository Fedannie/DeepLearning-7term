# Second assignment

## Task: Implement ResNext
Since we still didn't settle with computational power I'll try to dedicate this homework mostly to code style practies in designing neural networks and doing so particulary in pytorch.

## Requirements:

- NO JUPYTER! Homework should be a valid pip package. See https://the-hitchhikers-guide-to-packaging.readthedocs.io/en/latest/quickstart.html, steps 1-3
- Add at least one test using forward pass with random data You can find this useful https://docs.pytest.org/en/latest/
- Architecure should be similar to torchvision/models/resnet.py You are basically asked to replace ResNet residual block with the block of your own
- I suggest inplementing Trainer class, being able to train network regardless of the architecure
- Trainer should provide loss logging into tensorboardX
- No stdout, only logging
- Model should be able to load from the pretrained weights
- For architecture reference visit https://arxiv.org/abs/1611.05431
- I'm okay with you searching on the github for source code examples and best practices, but I sure will notice lazy copy-paste :)
