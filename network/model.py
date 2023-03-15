import torch
import torch.nn as nn
from collections import OrderedDict
from functools import reduce


class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32, image_size=64):
        super(UNet, self).__init__()

        cur_size = image_size

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        if (cur_size % 2 == 0):
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
        else:
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2, output_padding=1
            )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")
        cur_size = cur_size//2

        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        if (cur_size % 2 == 0):
            self.upconv2 = nn.ConvTranspose2d(
                features * 4, features*2, kernel_size=2, stride=2
            )
        else:
            self.upconv2 = nn.ConvTranspose2d(
                features * 4, features*2, kernel_size=2, stride=2, output_padding=1
            )
        self.decoder2 = UNet._block(
            (features * 2) * 2, features * 2, name="dec2")
        cur_size = cur_size//2

        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        if (cur_size % 2 == 0):
            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
        else:
            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2, output_padding=1
            )
        self.decoder3 = UNet._block(
            (features * 4) * 2, features * 4, name="dec3")
        cur_size = cur_size//2

        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        if (cur_size % 2 == 0):
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2
            )
        else:
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2, output_padding=1
            )
        self.decoder4 = UNet._block(
            (features * 8) * 2, features * 8, name="dec4")

        self.bottleneck = UNet._block(
            features * 8, features * 16, name="bottleneck")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
        self.weight_init()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv(dec1)

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


class UNetEncoder(nn.Module):
    def __init__(self, image_size, image_channel, features, downsample_factor, initial_feature_channels):
        super().__init__()
        self.image_size_downsampled = image_size//downsample_factor
        self.init_feature_channel = initial_feature_channels

        downsample_layers = []
        count = 0
        downsample_layers.append((f'downsample_pad_{count}',
                                  nn.ReflectionPad2d(3)))
        downsample_layers.append((f'downsample_conv_{count}',
                                  nn.Conv2d(image_channel, self.init_feature_channel,
                                            kernel_size=7, padding=0)))
        downsample_layers.append((f'downsample_bn_{count}',
                                  nn.BatchNorm2d(self.init_feature_channel)))
        downsample_layers.append((f'downsample_act_{count}',
                                  nn.ReLU(True)))

        downsample_layers += self.build_downsample_layers(downsample_factor)

        self.downsample_layers = nn.Sequential(OrderedDict(downsample_layers))

        self.encoder = UNet(self.init_feature_channel,
                            self.init_feature_channel,
                            self.init_feature_channel, self.image_size_downsampled)
        self.feature_layers = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(self.image_size_downsampled * self.image_size_downsampled *
                      self.init_feature_channel, features),
            nn.ReLU(inplace=True),
            nn.Linear(features, features)
        )

    def build_downsample_layers(self, downsample_factor):
        df = downsample_factor
        downsample_layers = []
        count = 1
        while (df != 1):
            downsample_layers.append((f'downsample_conv_{count}',
                                      nn.Conv2d(self.init_feature_channel,
                                                self.init_feature_channel, 3, 2, 1)))
            downsample_layers.append((f'downsample_bn_{count}',
                                      nn.BatchNorm2d(self.init_feature_channel)))
            downsample_layers.append((f'downsample_act_{count}',
                                      nn.ReLU(True)))
            df //= 2
            count += 1
        return downsample_layers

    def forward(self, x):
        if self.downsample_layers is not None:
            x = self.downsample_layers(x)
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.feature_layers(x)
        return x


class UNetDecoder(nn.Module):
    def __init__(self, image_size, image_channel, features, upsample_factor, initial_feature_channels):
        super().__init__()
        self.image_size = image_size
        self.image_channel = image_channel

        self.image_size_downsampled = image_size//upsample_factor
        self.init_feature_channel = initial_feature_channels

        upsample_layers = self.build_upsample_layers(upsample_factor)

        upsample_layers.append(('last_pad', nn.ReflectionPad2d(3)))
        upsample_layers.append(('last_conv',
                               nn.Conv2d(self.init_feature_channel,
                                         self.image_channel, kernel_size=7, padding=0)))
        self.upsample_layers = nn.Sequential(OrderedDict(upsample_layers))

        self.decoder = UNet(self.init_feature_channel,
                            self.init_feature_channel,
                            self.init_feature_channel,
                            self.image_size_downsampled)
        self.feature_layers = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(features, self.image_size_downsampled *
                      self.image_size_downsampled * self.init_feature_channel),
            nn.ReLU(inplace=True),
        )

    def build_upsample_layers(self, upsample_factor):
        uf = upsample_factor
        upsample_layers = []
        count = 0
        while (uf != 1):
            upsample_layers.append((f'downsample_conv_{count}',
                                    nn.ConvTranspose2d(self.init_feature_channel,
                                                       self.init_feature_channel, 2, 2)))
            upsample_layers.append((f'downsample_bn_{count}',
                                    nn.BatchNorm2d(self.init_feature_channel)))
            upsample_layers.append((f'downsample_act_{count}',
                                    nn.ReLU(True)))
            uf //= 2
            count += 1

        return upsample_layers

    def forward(self, x):
        x = self.feature_layers(x)
        x = x.view(x.size(0), self.init_feature_channel,
                   self.image_size_downsampled, self.image_size_downsampled)
        x = self.decoder(x)
        x = self.upsample_layers(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, image_size, image_channel, features, downsample_factor, initial_feature_channels, arch):
        super().__init__()
        self.arch = arch
        if arch.startswith('unet'):
            self.encoder = UNet(image_channel, image_channel,
                                initial_feature_channels)
            self.decoder = lambda x: x
        elif arch.startswith('wnet'):
            self.encoder = UNetEncoder(
                image_size, image_channel, features, downsample_factor, initial_feature_channels)
            self.decoder = UNetDecoder(
                image_size, image_channel, features, downsample_factor, initial_feature_channels)
        else:
            raise RuntimeError()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
