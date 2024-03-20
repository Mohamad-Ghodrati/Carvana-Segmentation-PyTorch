import torch
import torch.nn as nn
import torchvision.transforms.functional as TF 

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(self, 
                 in_channels=3,
                 out_channels=1,
                 features=[64, 128, 256, 512]):
        
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Down part of U-Net
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        self.bottleneck = DoubleConv(features[-1], 2*features[-1])
        
        # Up part of U-Net
        for feature in features[::-1]:
            self.ups.append(nn.ConvTranspose2d(2*feature, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(2*feature, feature))
            
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
    def forward(self, x):
        skip_connections = []
        
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
                
        skip_connections = skip_connections[::-1]
        
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            
            if x.shape != skip_connection.shape:
                x = TF.resize(x, skip_connection.shape[2:])
            
            x = torch.cat((skip_connection, x), axis=1)  
            x = self.ups[idx+1](x)

        return self.final_conv(x)
                        


if __name__ == '__main__':
    model = UNET()
    random_input = torch.randn((1, 3, 100, 100))
    output = model(random_input)
    print(f'{output.shape = }')