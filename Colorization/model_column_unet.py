import torch
import torch.nn as nn
from torchvision import models

class ScaleLayer(nn.Module):

   def __init__(self, init_value=1e-3):
       super().__init__()
       self.scale = nn.Parameter(torch.FloatTensor([init_value]))

   def forward(self, input):
       return input * self.scale

def weights_init(model):
    if type(model) in [nn.Conv2d, nn.Linear]:
        nn.init.xavier_normal_(model.weight.data)
        # nn.init.constant_(model.bias.data, 0.1)

class Color_model(nn.Module):
    def __init__(self):
        super(Color_model, self).__init__()

        # conv1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features = 64),
        )
        self.conv1_2norm_ss = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 1, stride = 2, groups = 64,
                                        bias = False)

        # conv2
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features = 128),
        )
        self.conv2_2norm_ss = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 1, stride = 2, groups = 128, bias = False)

        # conv3
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features = 256),
        )
        self.conv3_3norm_ss = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 1, stride = 2, groups = 256, bias = False)

        # conv4
        self.conv4=nn.Sequential(
            nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features = 512),
        )

        # conv5
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 2, dilation = 2),
            nn.ReLU(),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 2, dilation = 2),
            nn.ReLU(),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 2, dilation = 2),
            nn.ReLU(),
            nn.BatchNorm2d(num_features = 512),
        )

        # conv6
        self.conv6 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 2, dilation = 2),
            nn.ReLU(),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 2, dilation = 2),
            nn.ReLU(),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 2, dilation = 2),
            nn.ReLU(),
            nn.BatchNorm2d(num_features = 512),
        )

        # conv7
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1, dilation = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1, dilation = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1, dilation = 1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features = 512),
        )

        # conv8
        self.conv8_1 = nn.ConvTranspose2d(in_channels = 512, out_channels = 256, kernel_size = 4, stride = 2, padding = 1, dilation = 1)
        self.conv3_3_short = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1,stride = 1)
        self.conv8_sub = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1, dilation = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1, dilation = 1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features = 256),
        )

        # conv9
        # self.conv9_1 = nn.ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size = 4, stride = 2,padding = 1, dilation = 1)
        # self.conv2_2_short = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1, stride = 1)
        # self.conv9_sub = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1, dilation = 1),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(num_features = 128),
        # )

        self.conv3_pred = nn.Conv2d(in_channels = 256, out_channels = 384, kernel_size = 3, stride = 1, padding = 1, dilation = 1)
        self.conv4_pred = nn.ConvTranspose2d(in_channels = 512, out_channels = 384, kernel_size = 4, stride = 2, padding = 1, dilation = 1)
        self.conv5_pred = nn.ConvTranspose2d(in_channels = 512, out_channels = 384, kernel_size = 4, stride = 2, padding = 1, dilation = 1)
        self.conv6_pred = nn.ConvTranspose2d(in_channels = 512, out_channels = 384, kernel_size = 4, stride = 2, padding = 1, dilation = 1)
        self.conv7_pred = nn.ConvTranspose2d(in_channels = 512, out_channels = 384, kernel_size = 4, stride = 2, padding = 1, dilation = 1)
        self.conv8_pred = nn.Conv2d(in_channels = 256, out_channels = 384, kernel_size = 3, stride =1, padding = 1, dilation = 1)

        self.relu_313 = nn.ReLU()
        self.pred_313 = nn.Conv2d(in_channels = 384, out_channels = 313, kernel_size = 1, stride = 1, dilation = 1)
        self.pred_313_us = nn.ConvTranspose2d(in_channels = 313, out_channels = 313, kernel_size = 4, stride = 2, padding = 1, groups = 313)
        # # conv10
        # self.conv10_1 = nn.ConvTranspose2d(in_channels = 128, out_channels = 128, kernel_size = 4, stride = 2, padding = 1, dilation = 1)
        # self.conv1_2_short = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1, stride = 1)
        # self.conv10_sub = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1, dilation = 1),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels = 128, out_channels = 313, kernel_size = 1, stride = 1,dilation = 1),
        # )

        self.apply(weights_init)

    def forward(self, gray_image):
        conv1 = self.conv1(gray_image)
        conv1_2norm_ss = self.conv1_2norm_ss(conv1)
        conv2 = self.conv2(conv1_2norm_ss)
        conv2_2norm_ss = self.conv2_2norm_ss(conv2)
        conv3 = self.conv3(conv2_2norm_ss)
        conv3_3norm_ss = self.conv3_3norm_ss(conv3)
        conv4 = self.conv4(conv3_3norm_ss)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)
        conv7 = self.conv7(conv6)
        conv8_1 = self.conv8_1(conv7)
        conv3_3_short = self.conv3_3_short(conv3)
        conv8_1_comb = conv8_1 + conv3_3_short
        conv8_sub = self.conv8_sub(conv8_1_comb)

        conv3_pred = self.conv3_pred(conv3)
        conv4_pred = self.conv4_pred(conv4)
        conv5_pred = self.conv5_pred(conv5)
        conv6_pred = self.conv6_pred(conv6)
        conv7_pred = self.conv7_pred(conv7)
        conv8_pred = self.conv8_pred(conv8_sub)
        conv345678_pred = conv3_pred+conv4_pred+conv5_pred+conv6_pred+conv7_pred+conv8_pred
        relu345678_pred = self.relu_313(conv345678_pred)
        pred_313 = self.pred_313(relu345678_pred)
        features =self.pred_313_us(pred_313)
        # features = self.conv10_sub(conv10_comb)
        features=features/0.38
        return features
