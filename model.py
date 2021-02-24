import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import BatchNorm2d
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),  # Applies a same convolution (height, width in = height, width out)
            nn.BatchNorm2d(out_channels),                               # We're using bias=False because we're using batch norm
            nn.ReLU(inplace=True),                                      # inplace=True: modifies the input directly
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x) 


class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()                      # List that can store conv layers etc
        self.downs = nn.ModuleList()                    # This is important to be able to use model.eval etc
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:    
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                # This part doubles the height and the width of the image
                nn.ConvTranspose2d(
                    in_channels=feature*2, out_channels=feature, kernel_size=2, stride=2
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        # The part at the bottom, from here on out refered to as the bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        # The last conv layer at the end of the UNET model
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]   # Reverses the skip_connections list


        for idx in range(0, len(self.ups), 2):
            """
                We're doing one convTranspose and then the double conv pr step.
                That's why we're doing a step of two here, since the structure of ups is

                ups: [
                    conv_transpose,
                    doubleConv,
                    conv_transpose,
                    doubleConv,
                    ...
                ]
            """

            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]                  # We have to do this because we're doing steps of 2
            
            if x.shape != skip_connection.shape:
                """
                    MaxPool will always floor the division. Meaning if we have an input
                    of 161x161, the result will be 80x80. When we upsample this, we'll
                    get a 160x160 image, meaning we won't be able to concat them. This is why
                    we can resize the x if it doesn't match the size of the skip_connect.

                    TODO: Look into if this impacts the accuracy
                """
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)        # We're adding them along the channel dimension
            x = self.ups[idx+1](concat_skip)                            # Performes the DoubleConv on the concat of x and the skip
        
        return self.final_conv(x)

def test():
    x = torch.randn((3, 1, 161, 161))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()