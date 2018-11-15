import torch.nn as nn


class DCGenerator(nn.Module):
    kernel = 3
    stride = 1
    padding = 1
    
    def __init__(self, image_size, channels, data_size):
        super(DCGenerator, self).__init__()

        self.init_size = image_size // 8
        self.final_image_size = image_size * 8

        self.linear = nn.Sequential(nn.Linear(data_size, image_size * 8 * self.init_size ** 2))

        self.main = nn.Sequential(
            nn.BatchNorm2d(image_size * 8),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(image_size * 8, image_size * 4, self.kernel, self.stride, self.padding),
            nn.BatchNorm2d(image_size * 4, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(image_size * 4, image_size * 2, self.kernel, self.stride, self.padding),
            nn.BatchNorm2d(image_size * 2, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(image_size * 2, image_size, self.kernel, self.stride, self.padding),
            nn.BatchNorm2d(image_size, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(image_size, channels, self.kernel, self.stride, self.padding),
            nn.Tanh()
        )

    def forward(self, data):
        x = data.view(data.shape[0], -1)
        x = self.linear(x)
        x = x.view(x.shape[0], self.final_image_size, self.init_size, self.init_size)
        return self.main(x)

class DCDiscriminator(nn.Module):
    kernel = 3
    stride = 2
    padding = 1
    
    def __init__(self, image_size, channels):
        super(DCDiscriminator, self).__init__()
        size = image_size // 2 ** 4
        self.main = nn.Sequential(
            nn.Conv2d(channels, image_size, self.kernel, self.stride, self.padding, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(image_size, image_size * 2, self.kernel, self.stride, self.padding, bias=False),
            nn.BatchNorm2d(image_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(image_size * 2, image_size * 4, self.kernel, self.stride, self.padding, bias=False),
            nn.BatchNorm2d(image_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(image_size * 4, image_size * 8, self.kernel, self.stride, self.padding, bias=False),
            nn.BatchNorm2d(image_size * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.linear = nn.Sequential(
            nn.Linear(image_size * 8 * size ** 2, 1),
            nn.Sigmoid())

    def forward(self, data):
        x = self.main(data)
        x = x.view(x.shape[0], -1)
        return self.linear(x)
