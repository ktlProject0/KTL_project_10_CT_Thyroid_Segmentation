import torch
import torch.nn as nn

class DenseBlock2D(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers, use_batchnorm=True):
        super(DenseBlock2D, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            layer_in_channels = in_channels + i * growth_rate
            self.layers.append(Conv2dLayer(layer_in_channels, growth_rate, use_batchnorm))

    def forward(self, x):
        for layer in self.layers:
            out = layer(x)
            x = torch.cat([x, out], dim=1)
        return x

class Conv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super(Conv2dLayer, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ]
        if use_batchnorm:
            layers.insert(1, nn.BatchNorm2d(out_channels))
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class TransitionDown2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionDown2D, self).__init__()
        self.transition = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.transition(x)

class TransitionUp2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionUp2D, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.upconv(x)

class Net(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32, growth_rate=16, n_dense_layers=4):
        super(Net, self).__init__()
        self.init_conv = nn.Conv2d(in_channels, init_features, kernel_size=3, padding=1)
        self.dense_block1 = DenseBlock2D(init_features, growth_rate, n_dense_layers)
        self.trans_down1 = TransitionDown2D(init_features + n_dense_layers * growth_rate, init_features * 2)
        self.dense_block2 = DenseBlock2D(init_features * 2, growth_rate, n_dense_layers)
        self.trans_down2 = TransitionDown2D(init_features * 2 + n_dense_layers * growth_rate, init_features * 4)
        self.dense_block3 = DenseBlock2D(init_features * 4, growth_rate, n_dense_layers)
        self.trans_down3 = TransitionDown2D(init_features * 4 + n_dense_layers * growth_rate, init_features * 8)

        self.bottleneck = DenseBlock2D(init_features * 8, growth_rate, n_dense_layers)

        self.trans_up3 = TransitionUp2D(init_features * 8 + n_dense_layers * growth_rate, init_features * 4)
        self.reduce_conv3 = nn.Conv2d(init_features * 4 + init_features * 4 + n_dense_layers * growth_rate, init_features * 4, kernel_size=1)
        self.dense_block4 = DenseBlock2D(init_features * 4, growth_rate, n_dense_layers)

        self.trans_up2 = TransitionUp2D(init_features * 4 + n_dense_layers * growth_rate, init_features * 2)
        self.reduce_conv2 = nn.Conv2d(init_features * 2 + init_features * 2 + n_dense_layers * growth_rate, init_features * 2, kernel_size=1)
        self.dense_block5 = DenseBlock2D(init_features * 2, growth_rate, n_dense_layers)

        self.trans_up1 = TransitionUp2D(init_features * 2 + n_dense_layers * growth_rate, init_features)
        self.reduce_conv1 = nn.Conv2d(init_features + init_features + n_dense_layers * growth_rate, init_features, kernel_size=1)
        self.dense_block6 = DenseBlock2D(init_features, growth_rate, n_dense_layers)

        self.final_conv = nn.Conv2d(init_features + n_dense_layers * growth_rate, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.init_conv(x)
        x1 = self.dense_block1(x1)
        x2 = self.trans_down1(x1)

        x2 = self.dense_block2(x2)
        x3 = self.trans_down2(x2)

        x3 = self.dense_block3(x3)
        x4 = self.trans_down3(x3)

        x4 = self.bottleneck(x4)

        x4 = self.trans_up3(x4)
        x4 = torch.cat((x4, x3), dim=1)
        x4 = self.reduce_conv3(x4)
        x4 = self.dense_block4(x4)

        x3 = self.trans_up2(x4)
        x3 = torch.cat((x3, x2), dim=1)
        x3 = self.reduce_conv2(x3)
        x3 = self.dense_block5(x3)

        x2 = self.trans_up1(x3)
        x2 = torch.cat((x2, x1), dim=1)
        x2 = self.reduce_conv1(x2)
        x2 = self.dense_block6(x2)

        x_out = self.final_conv(x2)
        return self.sigmoid(x_out)
