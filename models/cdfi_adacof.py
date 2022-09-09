import torch
import cupy_module.adacof as adacof
import sys
from torch.nn import functional as F
import torch.nn as nn
from utility import CharbonnierFunc, moduleNormalize


def make_model(args):
    return CDFI_adacof(args)


class CDFI_adacof(torch.nn.Module):
    def __init__(self, args):
        super(CDFI_adacof, self).__init__()
        self.args = args
        self.kernel_size = args.kernel_size
        self.kernel_pad = int(((args.kernel_size - 1) * args.dilation) / 2.0)
        self.dilation = args.dilation

        self.get_kernel = PrunedKernelEstimation(self.kernel_size)

        self.context_synthesis = GridNet(
            126, 3
        )  # (in_channel, out_channel) = (126, 3) for the synthesis network

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
        (
            Weight1,
            Alpha1,
            Beta1,
            Weight2,
            Alpha2,
            Beta2,
            Occlusion,
            Blend,
            featConv1,
            featConv2,
            featConv3,
            featConv4,
            featConv5,
        ) = self.get_kernel(moduleNormalize(frame0), moduleNormalize(frame2))

        tensorAdaCoF1 = (
            self.moduleAdaCoF(
                self.modulePad(frame0), Weight1, Alpha1, Beta1, self.dilation
            )
            * 1.0
        )
        tensorAdaCoF2 = (
            self.moduleAdaCoF(
                self.modulePad(frame2), Weight2, Alpha2, Beta2, self.dilation
            )
            * 1.0
        )

        w, h = self.modulePad(frame0).shape[2:]

        tensorConv1_ = F.interpolate(
            featConv1, size=(w, h), mode="bilinear", align_corners=False
        )
        tensorConv1L = (
            self.moduleAdaCoF(tensorConv1_, Weight1, Alpha1, Beta1, self.dilation) * 1.0
        )
        tensorConv1R = (
            self.moduleAdaCoF(tensorConv1_, Weight2, Alpha2, Beta2, self.dilation) * 1.0
        )

        tensorConv2_ = F.interpolate(
            featConv2, size=(w, h), mode="bilinear", align_corners=False
        )
        tensorConv2L = (
            self.moduleAdaCoF(tensorConv2_, Weight1, Alpha1, Beta1, self.dilation) * 1.0
        )
        tensorConv2R = (
            self.moduleAdaCoF(tensorConv2_, Weight2, Alpha2, Beta2, self.dilation) * 1.0
        )

        tensorConv3_ = F.interpolate(
            featConv3, size=(w, h), mode="bilinear", align_corners=False
        )
        tensorConv3L = (
            self.moduleAdaCoF(tensorConv3_, Weight1, Alpha1, Beta1, self.dilation) * 1.0
        )
        tensorConv3R = (
            self.moduleAdaCoF(tensorConv3_, Weight2, Alpha2, Beta2, self.dilation) * 1.0
        )

        tensorConv4_ = F.interpolate(
            featConv4, size=(w, h), mode="bilinear", align_corners=False
        )
        tensorConv4L = (
            self.moduleAdaCoF(tensorConv4_, Weight1, Alpha1, Beta1, self.dilation) * 1.0
        )
        tensorConv4R = (
            self.moduleAdaCoF(tensorConv4_, Weight2, Alpha2, Beta2, self.dilation) * 1.0
        )

        tensorConv5_ = F.interpolate(
            featConv5, size=(w, h), mode="bilinear", align_corners=False
        )
        tensorConv5L = (
            self.moduleAdaCoF(tensorConv5_, Weight1, Alpha1, Beta1, self.dilation) * 1.0
        )
        tensorConv5R = (
            self.moduleAdaCoF(tensorConv5_, Weight2, Alpha2, Beta2, self.dilation) * 1.0
        )

        tensorCombined = torch.cat(
            [
                tensorAdaCoF1,
                tensorAdaCoF2,
                tensorConv1L,
                tensorConv1R,
                tensorConv2L,
                tensorConv2R,
                tensorConv3L,
                tensorConv3R,
                tensorConv4L,
                tensorConv4R,
                tensorConv5L,
                tensorConv5R,
            ],
            dim=1,
        )

        frame1_warp = Occlusion * tensorAdaCoF1 + (1 - Occlusion) * tensorAdaCoF2
        frame1_feat = self.context_synthesis(tensorCombined)

        frame1 = (
            Blend * frame1_feat + (1 - Blend) * frame1_warp
        )  # blending of the feature warp and ordinary warp

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

            g_Spatial = (
                g_Alpha1 + g_Beta1 + g_Alpha2 + g_Beta2
            )  # used as total variation loss during training

            return {"frame1": frame1, "g_Spatial": g_Spatial}
        else:
            return frame1


class GridNet(nn.Module):
    def __init__(self, in_chs, out_chs, grid_chs=(32, 64, 96)):
        super(GridNet, self).__init__()

        self.n_row = 3
        self.n_col = 6
        self.n_chs = grid_chs
        assert (
            len(grid_chs) == self.n_row
        ), "should give num channels for each row (scale stream)"

        self.lateral_init = LateralBlock(in_chs, self.n_chs[0])

        for r, n_ch in enumerate(self.n_chs):
            for c in range(self.n_col - 1):
                setattr(self, f"lateral_{r}_{c}", LateralBlock(n_ch, n_ch))

        for r, (in_ch, out_ch) in enumerate(zip(self.n_chs[:-1], self.n_chs[1:])):
            for c in range(int(self.n_col / 2)):
                setattr(self, f"down_{r}_{c}", DownSamplingBlock(in_ch, out_ch))

        for r, (in_ch, out_ch) in enumerate(zip(self.n_chs[1:], self.n_chs[:-1])):
            for c in range(int(self.n_col / 2)):
                setattr(self, f"up_{r}_{c}", UpSamplingBlock(in_ch, out_ch))

        self.lateral_final = LateralBlock(self.n_chs[0], out_chs)

    def forward(self, x):
        state_00 = self.lateral_init(x)
        state_10 = self.down_0_0(state_00)
        state_20 = self.down_1_0(state_10)

        state_01 = self.lateral_0_0(state_00)
        state_11 = self.down_0_1(state_01) + self.lateral_1_0(state_10)
        state_21 = self.down_1_1(state_11) + self.lateral_2_0(state_20)

        state_02 = self.lateral_0_1(state_01)
        state_12 = self.down_0_2(state_02) + self.lateral_1_1(state_11)
        state_22 = self.down_1_2(state_12) + self.lateral_2_1(state_21)

        state_23 = self.lateral_2_2(state_22)
        state_13 = self.up_1_0(state_23) + self.lateral_1_2(state_12)
        state_03 = self.up_0_0(state_13) + self.lateral_0_2(state_02)

        state_24 = self.lateral_2_3(state_23)
        state_14 = self.up_1_1(state_24) + self.lateral_1_3(state_13)
        state_04 = self.up_0_1(state_14) + self.lateral_0_3(state_03)

        state_25 = self.lateral_2_4(state_24)
        state_15 = self.up_1_2(state_25) + self.lateral_1_4(state_14)
        state_05 = self.up_0_2(state_15) + self.lateral_0_4(state_04)

        return self.lateral_final(state_05)


class LateralBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(LateralBlock, self).__init__()
        self.f = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1),
        )
        if ch_in != ch_out:
            self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1)

    def forward(self, x):
        fx = self.f(x)
        if fx.shape[1] != x.shape[1]:
            x = self.conv(x)
        return fx + x


class DownSamplingBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(DownSamplingBlock, self).__init__()
        self.f = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=2, padding=1),
            nn.PReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.f(x)


class UpSamplingBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(UpSamplingBlock, self).__init__()
        self.f = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.PReLU(),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.f(x)


class PrunedKernelEstimation(torch.nn.Module):
    def __init__(self, kernel_size):
        super(PrunedKernelEstimation, self).__init__()
        self.kernel_size = kernel_size

        self.module1by1_1 = torch.nn.Conv2d(
            in_channels=24, out_channels=4, kernel_size=1, stride=1, padding=1
        )
        self.module1by1_2 = torch.nn.Conv2d(
            in_channels=52, out_channels=8, kernel_size=1, stride=1, padding=1
        )
        self.module1by1_3 = torch.nn.Conv2d(
            in_channels=95, out_channels=12, kernel_size=1, stride=1, padding=1
        )
        self.module1by1_4 = torch.nn.Conv2d(
            in_channels=159, out_channels=16, kernel_size=1, stride=1, padding=1
        )
        self.module1by1_5 = torch.nn.Conv2d(
            in_channels=121, out_channels=20, kernel_size=1, stride=1, padding=1
        )

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
                in_channels=48, out_channels=52, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
        )
        self.modulePool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv3 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=52, out_channels=99, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=99, out_channels=97, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=97, out_channels=95, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
        )
        self.modulePool3 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv4 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=95, out_channels=156, kernel_size=3, stride=1, padding=1
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
                in_channels=72, out_channels=95, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
        )

        self.moduleDeconv2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=95, out_channels=45, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=45, out_channels=45, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=45, out_channels=45, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
        )
        self.moduleUpsample2 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            torch.nn.Conv2d(
                in_channels=45, out_channels=52, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
        )

        self.moduleWeight1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=52, out_channels=52, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=52, out_channels=49, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=49, out_channels=21, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            torch.nn.Conv2d(
                in_channels=21, out_channels=121, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.Softmax(dim=1),
        )
        self.moduleAlpha1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=52, out_channels=52, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=52, out_channels=49, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=49, out_channels=21, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            torch.nn.Conv2d(
                in_channels=21, out_channels=121, kernel_size=3, stride=1, padding=1
            ),
        )
        self.moduleBeta1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=52, out_channels=52, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=52, out_channels=49, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=49, out_channels=21, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            torch.nn.Conv2d(
                in_channels=21, out_channels=121, kernel_size=3, stride=1, padding=1
            ),
        )
        self.moduleWeight2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=52, out_channels=52, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=52, out_channels=49, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=49, out_channels=21, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            torch.nn.Conv2d(
                in_channels=21, out_channels=121, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.Softmax(dim=1),
        )
        self.moduleAlpha2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=52, out_channels=52, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=52, out_channels=49, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=49, out_channels=21, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            torch.nn.Conv2d(
                in_channels=21, out_channels=121, kernel_size=3, stride=1, padding=1
            ),
        )
        self.moduleBeta2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=52, out_channels=52, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=52, out_channels=49, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=49, out_channels=21, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            torch.nn.Conv2d(
                in_channels=21, out_channels=121, kernel_size=3, stride=1, padding=1
            ),
        )
        self.moduleOcclusion = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=52, out_channels=52, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=52, out_channels=52, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=52, out_channels=48, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            torch.nn.Conv2d(
                in_channels=48, out_channels=1, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.Sigmoid(),
        )
        self.moduleBlend = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=52, out_channels=52, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=52, out_channels=52, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=52, out_channels=48, kernel_size=3, stride=1, padding=1
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
        Blend = self.moduleBlend(tensorCombine)

        featConv1 = self.module1by1_1(tensorConv1)
        featConv2 = self.module1by1_2(tensorConv2)
        featConv3 = self.module1by1_3(tensorConv3)
        featConv4 = self.module1by1_4(tensorConv4)
        featConv5 = self.module1by1_5(tensorConv5)

        return (
            Weight1,
            Alpha1,
            Beta1,
            Weight2,
            Alpha2,
            Beta2,
            Occlusion,
            Blend,
            featConv1,
            featConv2,
            featConv3,
            featConv4,
            featConv5,
        )
