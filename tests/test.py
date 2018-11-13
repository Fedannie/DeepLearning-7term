import torch

from new_resnext.ResNextBlock import resnext50


def test():
    """
    Simple test on random data.
    """
    num_classes = 10
    batch_size = 8
    tensor = torch.rand(batch_size, 3, 224, 224)
    net = resnext50(num_classes=num_classes)
    output = net(tensor)
    assert output.size() == (batch_size, num_classes), "Test failed"
