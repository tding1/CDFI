import torch
import cupy_module.adacof as adacof
import sys
from torch.nn import functional as F
from utility import CharbonnierFunc, moduleNormalize


def make_model(args):
    return AdaCoFNet(args)


class AdaCoFNet(torch.nn.Module):
    def __init__(self, args):
        super(AdaCoFNet, self).__init__()
        self.args = args
        self.kernel_size = args.kernel_size
        self.kernel_pad = int(((args.kernel_size - 1) * args.dilation) / 2.0)
        self.dilation = args.dilation

        self.get_kernel = PrunedKernelEstimation(self.kernel_size)

        self.modulePad = torch.nn.ReplicationPad2d(
            [self.kernel_pad, self.kernel_pad, self.kernel_pad, self.kernel_pad]
        )

        self.moduleAdaCoF = adacof.FunctionAdaCoF.apply

    def forward(self, frame0, frame2):
        h0 = int(list(frame0.size())[2])
        w0 = int(list(frame0.size())[3])
        h2 = int(list(frame2.size())[2])
        w2 = int(list(frame2.size())[3])
        if h0 != h2 or w0 != w2:
            sys.exit("Frame sizes do not match")

        h_padded = False
        w_padded = False
        if h0 % 32 != 0:
            pad_h = 32 - (h0 % 32)
            frame0 = F.pad(frame0, [0, 0, 0, pad_h], mode="reflect")
            frame2 = F.pad(frame2, [0, 0, 0, pad_h], mode="reflect")
            h_padded = True

        if w0 % 32 != 0:
            pad_w = 32 - (w0 % 32)
            frame0 = F.pad(frame0, [0, pad_w, 0, 0], mode="reflect")
            frame2 = F.pad(frame2, [0, pad_w, 0, 0], mode="reflect")
            w_padded = True
        Weight1, Alpha1, Beta1, Weight2, Alpha2, Beta2, Occlusion = self.get_kernel(
            moduleNormalize(frame0), moduleNormalize(frame2)
        )

        tensorAdaCoF1 = self.moduleAdaCoF(
            self.modulePad(frame0), Weight1, Alpha1, Beta1, self.dilation
        )
        tensorAdaCoF2 = self.moduleAdaCoF(
            self.modulePad(frame2), Weight2, Alpha2, Beta2, self.dilation
        )

        frame1 = Occlusion * tensorAdaCoF1 + (1 - Occlusion) * tensorAdaCoF2
        if h_padded:
            frame1 = frame1[:, :, 0:h0, :]
        if w_padded:
            frame1 = frame1[:, :, :, 0:w0]

        if self.training:
            # Smoothness Terms
            m_Alpha1 = torch.mean(Weight1 * Alpha1, dim=1, keepdim=True)
            m_Alpha2 = torch.mean(Weight2 * Alpha2, dim=1, keepdim=True)
            m_Beta1 = torch.mean(Weight1 * Beta1, dim=1, keepdim=True)
            m_Beta2 = torch.mean(Weight2 * Beta2, dim=1, keepdim=True)

            g_Alpha1 = CharbonnierFunc(
                m_Alpha1[:, :, :, :-1] - m_Alpha1[:, :, :, 1:]
            ) + CharbonnierFunc(m_Alpha1[:, :, :-1, :] - m_Alpha1[:, :, 1:, :])
            g_Beta1 = CharbonnierFunc(
                m_Beta1[:, :, :, :-1] - m_Beta1[:, :, :, 1:]
            ) + CharbonnierFunc(m_Beta1[:, :, :-1, :] - m_Beta1[:, :, 1:, :])
            g_Alpha2 = CharbonnierFunc(
                m_Alpha2[:, :, :, :-1] - m_Alpha2[:, :, :, 1:]
            ) + CharbonnierFunc(m_Alpha2[:, :, :-1, :] - m_Alpha2[:, :, 1:, :])
            g_Beta2 = CharbonnierFunc(
                m_Beta2[:, :, :, :-1] - m_Beta2[:, :, :, 1:]
            ) + CharbonnierFunc(m_Beta2[:, :, :-1, :] - m_Beta2[:, :, 1:, :])
            g_Occlusion = CharbonnierFunc(
                Occlusion[:, :, :, :-1] - Occlusion[:, :, :, 1:]
            ) + CharbonnierFunc(Occlusion[:, :, :-1, :] - Occlusion[:, :, 1:, :])

            g_Spatial = g_Alpha1 + g_Beta1 + g_Alpha2 + g_Beta2

            return {
                "frame1": frame1,
                "g_Spatial": g_Spatial,
                "g_Occlusion": g_Occlusion,
            }
        else:
            return frame1


class PrunedKernelEstimation(torch.nn.Module):
    def __init__(self, kernel_size):
        super(PrunedKernelEstimation, self).__init__()
        self.kernel_size = kernel_size

        self.moduleConv1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=6, out_channels=24, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
        )
        self.modulePool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=24, out_channels=48, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=48, out_channels=51, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
        )
        self.modulePool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv3 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=51, out_channels=99, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=99, out_channels=97, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=97, out_channels=94, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
        )
        self.modulePool3 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv4 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=94, out_channels=156, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=156, out_channels=142, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=142, out_channels=159, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
        )
        self.modulePool4 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv5 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=159, out_channels=92, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=92, out_channels=72, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=72, out_channels=121, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
        )
        self.modulePool5 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleDeconv5 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=121, out_channels=99, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=99, out_channels=69, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=69, out_channels=36, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
        )
        self.moduleUpsample5 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            torch.nn.Conv2d(
                in_channels=36, out_channels=121, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
        )

        self.moduleDeconv4 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=121, out_channels=74, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=74, out_channels=83, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=83, out_channels=81, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
        )
        self.moduleUpsample4 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            torch.nn.Conv2d(
                in_channels=81, out_channels=159, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
        )

        self.moduleDeconv3 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=159, out_channels=83, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=83, out_channels=88, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=88, out_channels=72, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
        )
        self.moduleUpsample3 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            torch.nn.Conv2d(
                in_channels=72, out_channels=94, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
        )

        self.moduleDeconv2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=94, out_channels=45, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=45, out_channels=47, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=47, out_channels=44, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
        )
        self.moduleUpsample2 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            torch.nn.Conv2d(
                in_channels=44, out_channels=51, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
        )

        self.moduleWeight1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=51, out_channels=51, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=51, out_channels=49, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=49, out_channels=21, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            torch.nn.Conv2d(
                in_channels=21,
                out_channels=self.kernel_size**2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            torch.nn.Softmax(dim=1),
        )
        self.moduleAlpha1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=51, out_channels=50, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=50, out_channels=48, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=48, out_channels=20, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            torch.nn.Conv2d(
                in_channels=20,
                out_channels=self.kernel_size**2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
        self.moduleBeta1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=51, out_channels=51, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=51, out_channels=50, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=50, out_channels=20, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            torch.nn.Conv2d(
                in_channels=20,
                out_channels=self.kernel_size**2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
        self.moduleWeight2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=51, out_channels=50, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=50, out_channels=49, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=49, out_channels=20, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            torch.nn.Conv2d(
                in_channels=20,
                out_channels=self.kernel_size**2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            torch.nn.Softmax(dim=1),
        )
        self.moduleAlpha2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=51, out_channels=50, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=50, out_channels=49, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=49, out_channels=20, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            torch.nn.Conv2d(
                in_channels=20,
                out_channels=self.kernel_size**2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
        self.moduleBeta2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=51, out_channels=51, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=51, out_channels=50, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=50, out_channels=20, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            torch.nn.Conv2d(
                in_channels=20,
                out_channels=self.kernel_size**2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
        self.moduleOcclusion = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=51, out_channels=52, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=52, out_channels=51, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=51, out_channels=48, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            torch.nn.Conv2d(
                in_channels=48, out_channels=1, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.Sigmoid(),
        )

    def forward(self, rfield0, rfield2):
        tensorJoin = torch.cat([rfield0, rfield2], 1)

        tensorConv1 = self.moduleConv1(tensorJoin)
        tensorPool1 = self.modulePool1(tensorConv1)

        tensorConv2 = self.moduleConv2(tensorPool1)
        tensorPool2 = self.modulePool2(tensorConv2)

        tensorConv3 = self.moduleConv3(tensorPool2)
        tensorPool3 = self.modulePool3(tensorConv3)

        tensorConv4 = self.moduleConv4(tensorPool3)
        tensorPool4 = self.modulePool4(tensorConv4)

        tensorConv5 = self.moduleConv5(tensorPool4)
        tensorPool5 = self.modulePool5(tensorConv5)

        tensorDeconv5 = self.moduleDeconv5(tensorPool5)
        tensorUpsample5 = self.moduleUpsample5(tensorDeconv5)

        tensorCombine = tensorUpsample5 + tensorConv5

        tensorDeconv4 = self.moduleDeconv4(tensorCombine)
        tensorUpsample4 = self.moduleUpsample4(tensorDeconv4)

        tensorCombine = tensorUpsample4 + tensorConv4

        tensorDeconv3 = self.moduleDeconv3(tensorCombine)
        tensorUpsample3 = self.moduleUpsample3(tensorDeconv3)

        tensorCombine = tensorUpsample3 + tensorConv3

        tensorDeconv2 = self.moduleDeconv2(tensorCombine)
        tensorUpsample2 = self.moduleUpsample2(tensorDeconv2)

        tensorCombine = tensorUpsample2 + tensorConv2

        Weight1 = self.moduleWeight1(tensorCombine)
        Alpha1 = self.moduleAlpha1(tensorCombine)
        Beta1 = self.moduleBeta1(tensorCombine)
        Weight2 = self.moduleWeight2(tensorCombine)
        Alpha2 = self.moduleAlpha2(tensorCombine)
        Beta2 = self.moduleBeta2(tensorCombine)
        Occlusion = self.moduleOcclusion(tensorCombine)

        return Weight1, Alpha1, Beta1, Weight2, Alpha2, Beta2, Occlusion
