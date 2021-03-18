import torch
import torchvision
import torch.nn as nn
from torch.nn import functional as F

from utility import Module_CharbonnierLoss


class VGG_loss(nn.Module):
    def __init__(self):
        super(VGG_loss, self).__init__()
        vgg16 = torchvision.models.vgg16(pretrained=True)
        self.vgg16_conv_4_3 = torch.nn.Sequential(*list(vgg16.children())[0][:22])
        for param in self.vgg16_conv_4_3.parameters():
            param.requires_grad = False

    def forward(self, output, gt):
        vgg_output = self.vgg16_conv_4_3(output)
        with torch.no_grad():
            vgg_gt = self.vgg16_conv_4_3(gt.detach())

        loss = F.mse_loss(vgg_output, vgg_gt)

        return loss


class Loss(nn.modules.loss._Loss):
    def __init__(self, args):
        super(Loss, self).__init__()

        self.loss = []
        self.loss_module = nn.ModuleList()
        self.regularize = []

        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            elif loss_type == 'Charb':
                loss_function = Module_CharbonnierLoss()
            elif loss_type.find('VGG') >= 0:
                loss_function = VGG_loss()
            elif loss_type in ['g_Spatial', 'Lw', 'Ls']:
                self.regularize.append({
                    'type': loss_type,
                    'weight': float(weight)}
                )
                continue

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function}
            )

        print('Loss = ')
        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])

        for r in self.regularize:
            print('{:.3f} * {}'.format(r['weight'], r['type']))

        self.loss_module.to('cuda')

    def forward(self, output, gt):
        losses = []
        for l in self.loss:
            if l['function'] is not None:
                loss = l['function'](output['frame1'], gt)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)

        for r in self.regularize:
            effective_loss = r['weight'] * output[r['type']]
            losses.append(effective_loss)

        loss_sum = sum(losses)

        return loss_sum
