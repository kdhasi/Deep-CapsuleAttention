from __future__ import print_function

import argparse
import os

import torch.nn as nn
from prettytable import PrettyTable
from torch import optim

from datasets_util import *
from utils import *

parser = argparse.ArgumentParser(description='Parameters for deep hybrid networks.')
parser.add_argument('--dataset', type=str, default="cifar10", help="choose cifar10 or svhn dataset (default: cifar10)")
parser.add_argument('--class_num', type=int, default=10, help='number of classes for the used dataset (default: 10)')
parser.add_argument('--input_img_dim', type=int, default=3, help='number of image color channels (default: 3)')
parser.add_argument('--input_img_size', type=int, default=32, help='input image size (default: 32)')
parser.add_argument('--C', type=int, default=4, help='number of capsule channels (default: 4)')
parser.add_argument('--K', type=int, default=10, help='number of kernels used to form a cluster (default: 10)')
parser.add_argument('--D', type=int, default=32, help='capsule depth (default: 32)')
parser.add_argument('--if_bias', action='store_true', default=True, help='if use bias while transforming capsules')
parser.add_argument('--epochs', type=int, default=200, help='training epochs (default: 200)')
parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate (default: 0.1)')
parser.add_argument('--batch_size', type=int, default=64, help='training batch size (default: 64)')
parser.add_argument('--test_batch_size', type=int, default=128, help='testing batch size (default: 128)')
parser.add_argument('--gamma', type=float, default=0.1, help='decay rate for learning rate (default: 0.1)')
parser.add_argument('--L2_penalty_factor', type=float, default=0.0005, help='weight decay (default: 0.0005)')
parser.add_argument('--log_interval', type=int, default=350,
                    help='how many batches to wait before logging training status (default: 350)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='if disables CUDA training')
parser.add_argument('--seed', type=int, default=1, help='seed (default: 1)')
parser.add_argument('--working_dir', type=str, default="./", help="working directory.")
parser.add_argument('--step_size', type=int, default=75, help='step size to decay learning rate (default: 75)')
parser.add_argument('--save_dst', type=str, default="./checkpoint/", help="path to save best models.")
args = parser.parse_args()


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        params_str = "{:,}".format(params)
        table.add_row([name, params_str])
        total_params += params
    print(table)
    total_params_str = "{:,}".format(total_params)
    print(f"Total Trainable Params: {total_params_str}")
    return total_params


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        # 3D batch norm on channel dimension
        self.bn = nn.BatchNorm3d(out_channels, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class SpatialGate(nn.Module):
    def __init__(self, num_cluster, out_channels):
        super(SpatialGate, self).__init__()
        kernel_size = 3  # Use a kernel size of 3 for 3D convolutions in spatial attention
        self.num_cluster = num_cluster
        self.out_channels = out_channels
        self.compress = ChannelPool()
        self.spatial = BasicConv(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1,
                                 padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # Broadcasting
        return x * scale + x  # Add input to attended output


class RoutingWithSpatialAttention(nn.Module):
    def __init__(self, C_in, C_out, K, D_in, D_out, B, out_S, stride):
        super(RoutingWithSpatialAttention, self).__init__()
        self.K = K
        self.C_in = C_in
        self.C_out = C_out
        self.D_in = D_in
        self.D_out = D_out
        self.conv_trans = nn.ModuleList()
        self.value_conv = nn.ModuleList()
        for i in range(self.C_in):
            self.conv_trans.append(nn.Conv2d(self.D_in, self.C_out * self.D_out, 3,
                                             stride=stride, padding=1, bias=B))
        for i in range(self.C_out):
            self.value_conv.append(
                nn.Conv3d(in_channels=self.C_in, out_channels=self.K * self.C_in, kernel_size=(1, 1, 1),
                          stride=(1, 1, 1)))
        self.spatial_gate = SpatialGate(num_cluster=self.K, out_channels=self.C_out)
        self.acti = nn.LayerNorm([self.D_out, out_S, out_S])

    def cluster_routing(self, inter_votes):
        batch_size, _, h, w = inter_votes[0].shape
        for i in range(len(inter_votes)):
            inter_votes[i] = inter_votes[i].view(batch_size, self.C_out, self.D_out, h, w)
        votes_for_next_layer = []
        for i in range(self.C_out):
            to_cat = [inter_votes[j][:, i:(i + 1), :, :, :] for j in range(self.C_in)]
            votes_for_channel_i = torch.cat(to_cat, dim=1)
            votes_for_next_layer.append(votes_for_channel_i)

        values = [self.value_conv[i](votes_for_next_layer[i]) for i in range(self.C_out)]
        attended_values = [self.spatial_gate(value).view(batch_size, self.K, self.C_in, self.D_out, h, w) for value in
                           values]

        caps_of_next_layer = []
        for i in range(self.C_out):
            stds, means = torch.std_mean(attended_values[i], dim=1, unbiased=False)
            agreement = -torch.log(stds)
            atts_for_c1 = F.softmax(agreement, dim=1)
            caps_of_channel_i = (atts_for_c1 * means).sum(dim=1)
            caps_of_next_layer.append(caps_of_channel_i)

        return caps_of_next_layer

    def forward(self, caps):
        inter_votes = []
        for i in range(self.C_in):
            if isinstance(caps, list):  # if not first capsule layer
                inter_votes.append(self.conv_trans[i](caps[i]))
            else:  # for first capsule layer with image inputs
                inter_votes.append(self.conv_trans[i](caps))
        caps_of_next_layer = self.cluster_routing(inter_votes)
        for i in range(self.C_out):
            caps_of_next_layer[i] = self.acti(caps_of_next_layer[i])
        return caps_of_next_layer


class HybridNetwork(nn.Module):
    def __init__(self, args):
        super(HybridNetwork, self).__init__()
        self.caps_layer1 = RoutingWithSpatialAttention(args.C, args.C, args.K, args.input_img_dim, args.D, args.if_bias,
                                                       out_S=args.input_img_size, stride=1)
        self.caps_layer2 = RoutingWithSpatialAttention(args.C, args.C, args.K, args.D, args.D, args.if_bias,
                                                       out_S=args.input_img_size, stride=1)
        self.caps_layer3 = RoutingWithSpatialAttention(args.C, args.C, args.K, args.D, args.D, args.if_bias,
                                                       out_S=args.input_img_size, stride=1)

        self.caps_layer4 = RoutingWithSpatialAttention(args.C, args.C, args.K, args.D, args.D, args.if_bias,
                                                       out_S=int(args.input_img_size / 2), stride=2)
        self.caps_layer5 = RoutingWithSpatialAttention(args.C, args.C, args.K, args.D, args.D, args.if_bias,
                                                       out_S=int(args.input_img_size / 2), stride=1)
        self.caps_layer6 = RoutingWithSpatialAttention(args.C, args.C, args.K, args.D, args.D, args.if_bias,
                                                       out_S=int(args.input_img_size / 2), stride=1)

        self.caps_layer7 = RoutingWithSpatialAttention(args.C, args.C, args.K, args.D, args.D, args.if_bias,
                                                       out_S=int(args.input_img_size / 4), stride=2)
        self.caps_layer8 = RoutingWithSpatialAttention(args.C, args.C, args.K, args.D, args.D, args.if_bias,
                                                       out_S=int(args.input_img_size / 4), stride=1)
        self.caps_layer9 = RoutingWithSpatialAttention(args.C, args.C, args.K, args.D, args.D, args.if_bias,
                                                       out_S=int(args.input_img_size / 4), stride=1)

        self.caps_layer10 = RoutingWithSpatialAttention(args.C, args.C, args.K, args.D, args.D, args.if_bias,
                                                        out_S=int(args.input_img_size / 8), stride=2)
        self.caps_layer11 = RoutingWithSpatialAttention(args.C, args.C, args.K, args.D, args.D, args.if_bias,
                                                        out_S=int(args.input_img_size / 8), stride=1)
        self.caps_layer12 = RoutingWithSpatialAttention(args.C, args.class_num, args.K, args.D, args.D, args.if_bias,
                                                        out_S=int(args.input_img_size / 8), stride=1)

        self.classifier = nn.Linear(args.D * int(args.input_img_size / 8) ** 2, 1)

    def forward(self, x):
        caps_1 = self.caps_layer1(x)
        caps_2 = self.caps_layer2(caps_1)
        caps_3 = self.caps_layer3(caps_2)

        caps_4 = self.caps_layer4(caps_3)
        caps_5 = self.caps_layer5(caps_4)
        caps_6 = self.caps_layer6(caps_5)

        caps_7 = self.caps_layer7(caps_6)
        caps_8 = self.caps_layer8(caps_7)
        caps_9 = self.caps_layer9(caps_8)

        caps_10 = self.caps_layer10(caps_9)
        caps_11 = self.caps_layer11(caps_10)
        caps_12 = self.caps_layer12(caps_11)

        caps = [c.view(c.shape[0], -1).unsqueeze(1) for c in caps_12]
        caps = torch.cat(caps, dim=1)

        pred = self.classifier(caps).squeeze()
        return pred


def main(args):
    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = HybridNetwork(args).to(device)
    count_parameters(model)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.L2_penalty_factor)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    train_loader, test_loader, train_transform = get_dataset(args)

    if not os.path.exists(args.save_dst):
        os.makedirs(args.save_dst, exist_ok=False)

    best_acc = 0
    train_accuracies = []
    test_accuracies = []
    for epoch in range(0, args.epochs + 1):
        print('Current lr: {}\n'.format(scheduler.get_last_lr()))
        train_acc, train_loss = train(args, model, device, train_loader, optimizer, epoch)
        acc, loss = test(model, device, test_loader)
        scheduler.step()

        train_accuracies.append(train_acc)
        test_accuracies.append(acc)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(args.save_dst, args.dataset + "5L_best.pt"))
        print("Current training acc {:.3f}, test acc {:.3f}, best test acc {:.3f}\n".format(train_acc, acc, best_acc))


if __name__ == '__main__':
    main(args)
