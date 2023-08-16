import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


class Concatenate(nn.Module):
    def __init__(self):
        super(Concatenate, self).__init__()

    def forward(self, layer_1, layer_2):
        x = torch.cat((layer_1, layer_2), 1)
        return x


class Unet(nn.Module):
    def __init__(self, out_classes):
        super(Unet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32,
                               kernel_size=3, padding='same')

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=3, padding='same')

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=3, padding='same')

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256,
                               kernel_size=3, padding='same')

        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512,
                               kernel_size=3, padding='same')

        self.conv6 = nn.Conv2d(in_channels=512, out_channels=1024,
                               kernel_size=3, padding='same')

        self.conv7 = nn.Conv2d(in_channels=1024, out_channels=512,
                               kernel_size=3, padding='same')

        self.conv8 = nn.Conv2d(in_channels=512, out_channels=256,
                               kernel_size=3, padding='same')

        self.conv9 = nn.Conv2d(in_channels=256, out_channels=128,
                               kernel_size=3, padding='same')

        self.conv10 = nn.Conv2d(in_channels=128, out_channels=64,
                                kernel_size=3, padding='same')

        self.conv11 = nn.Conv2d(in_channels=64, out_channels=32,
                                kernel_size=3, padding='same')

        self.conv12 = nn.Conv2d(in_channels=32, out_channels=out_classes,
                                kernel_size=3, padding='same')

        self.conv13 = nn.Conv2d(in_channels=1024, out_channels=512,
                                kernel_size=3, padding='same')

        self.conv14 = nn.Conv2d(in_channels=512, out_channels=256,
                                kernel_size=3, padding='same')

        self.conv15 = nn.Conv2d(in_channels=256, out_channels=128,
                                kernel_size=3, padding='same')

        self.conv16 = nn.Conv2d(in_channels=128, out_channels=64,
                                kernel_size=3, padding='same')

        self.conv17 = nn.Conv2d(in_channels=64, out_channels=32,
                                kernel_size=3, padding='same')

        self.pool = nn.MaxPool2d(kernel_size=2)
        self.drop = nn.Dropout2d(p=0.2)
        self.up = nn.Upsample(scale_factor=2)
        self.concat = Concatenate()

    def forward(self, x):
        # stored = []

        # encoder 1
        x = self.conv1(x)
        store1 = F.relu(x)
        x = self.drop(x)
        x = self.pool(store1)

        # encoder 2
        x = self.conv2(x)
        store2 = F.relu(x)
        x = self.drop(x)
        x = self.pool(store2)

        # encoder 3
        x = self.conv3(x)
        store3 = F.relu(x)
        x = self.drop(x)
        x = self.pool(store3)

        # encoder 4
        x = self.conv4(x)
        store4 = F.relu(x)
        x = self.drop(x)
        x = self.pool(store4)

        # encoder 5
        x = self.conv5(x)
        store5 = F.relu(x)
        x = self.drop(x)
        x = self.pool(store5)

        x = self.conv6(x)
        x = self.drop(x)

        # decoder 1
        x = self.up(x)
        x = self.conv13(x)
        x = self.concat(x, store5)
        x = self.conv7(x)
        x = F.relu(x)
        x = self.drop(x)

        # decoder 2
        x = self.up(x)
        x = self.conv14(x)
        x = self.concat(x, store4)
        x = self.conv8(x)
        x = F.relu(x)
        x = self.drop(x)

        # decoder 3
        x = self.up(x)
        x = self.conv15(x)
        x = self.concat(x, store3)
        x = self.conv9(x)
        x = F.relu(x)
        x = self.drop(x)

        # decoder 4
        x = self.up(x)
        x = self.conv16(x)
        x = self.concat(x, store2)
        x = self.conv10(x)
        x = F.relu(x)
        x = self.drop(x)

        # decoder 5
        x = self.up(x)
        x = self.conv17(x)
        x = self.concat(x, store1)
        x = self.conv11(x)
        x = F.relu(x)
        x = self.drop(x)

        x = self.conv12(x)
        out = torch.sigmoid(x)
        return out
