export UNet
# define a UNet class in Julia
@pydef mutable struct UNet <: nn.Module
    function __init__(self, num_channels=3, num_classes=3)
        pybuiltin(:super)(UNet, self).__init__()
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv11 = nn.Conv1d(num_channels, 64, kernel_size=3, padding=1)
        self.conv12 = nn.Conv1d(64, 64, kernel_size=3, padding=1)

        self.conv21 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv22 = nn.Conv1d(128, 128, kernel_size=3, padding=1)

        self.conv31 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.conv32 = nn.Conv1d(256, 256, kernel_size=3, padding=1)

        self.conv41 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.conv42 = nn.Conv1d(512, 512, kernel_size=3, padding=1)

        self.conv51 = nn.Conv1d(512, 1024, kernel_size=3, padding=1)
        self.conv52 = nn.Conv1d(1024, 1024, kernel_size=3, padding=1)

        self.uconv6 = nn.ConvTranspose1d(1024, 512, kernel_size=2, stride=2)
        self.conv61 = nn.Conv1d(1024, 512, kernel_size=3, padding=1)
        self.conv62 = nn.Conv1d(512, 512, kernel_size=3, padding=1)

        self.uconv7 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        self.conv71 = nn.Conv1d(512, 256, kernel_size=3, padding=1)
        self.conv72 = nn.Conv1d(256, 256, kernel_size=3, padding=1)

        self.uconv8 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.conv81 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.conv82 = nn.Conv1d(128, 128, kernel_size=3, padding=1)

        self.uconv9 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.conv91 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.conv92 = nn.Conv1d(64, 64, kernel_size=3, padding=1)

        self.conv93 = nn.Conv1d(64, num_classes, kernel_size=1, padding=0)

        self.sigmoid = nn.Sigmoid()
    end

    function forward(self, x)
        x = self.conv11(x)
        x = self.relu(x)
        x = self.conv12(x)
        x1d = self.relu(x)
        x = self.maxpool(x1d)

        x = self.conv21(x)
        x = self.relu(x)
        x = self.conv22(x)
        x2d = self.relu(x)
        x = self.maxpool(x2d)

        x = self.conv31(x)
        x = self.relu(x)
        x = self.conv32(x)
        x3d = self.relu(x)
        x = self.maxpool(x3d)

        x = self.conv41(x)
        x = self.relu(x)
        x = self.conv42(x)
        x4d = self.relu(x)
        x = self.maxpool(x4d)

        x = self.conv51(x)
        x = self.relu(x)
        x = self.conv52(x)
        x5d = self.relu(x)

        x6u = self.uconv6(x5d)
        x = torch.cat((x4d, x6u), 1)
        x = self.conv61(x)
        x = self.relu(x)
        x = self.conv62(x)
        x = self.relu(x)

        x7u = self.uconv7(x)
        x = torch.cat((x3d, x7u), 1)
        x = self.conv71(x)
        x = self.relu(x)
        x = self.conv72(x)
        x = self.relu(x)

        x8u = self.uconv8(x)
        x = torch.cat((x2d, x8u), 1)
        x = self.conv81(x)
        x = self.relu(x)
        x = self.conv82(x)
        x = self.relu(x)

        x9u = self.uconv9(x)
        x = torch.cat((x1d, x9u), 1)
        x = self.conv91(x)
        x = self.relu(x)
        x = self.conv92(x)
        x = self.relu(x)

        x = self.conv93(x)

        x = self.sigmoid(x)
        #x = self.softmax(x)

        return x
    end
end
