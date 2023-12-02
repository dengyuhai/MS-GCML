import numpy as np
import torch
from collections import OrderedDict
import torch.nn as nn



class InceptionModule(nn.Module):
    def __init__(self, inplane, outplane_a1x1, outplane_b3x3_reduce, outplane_b3x3, outplane_c5x5_reduce, outplane_c5x5,
                 outplane_pool_proj):
        super(InceptionModule, self).__init__()
        self.a = nn.Sequential(OrderedDict([
            ('1x1', nn.Conv2d(inplane, outplane_a1x1, (1, 1), (1, 1), (0, 0))),
            ('1x1_relu', nn.ReLU(True))
        ]))

        self.b = nn.Sequential(OrderedDict([
            ('3x3_reduce', nn.Conv2d(inplane, outplane_b3x3_reduce, (1, 1), (1, 1), (0, 0))),
            ('3x3_relu1', nn.ReLU(True)),
            ('3x3', nn.Conv2d(outplane_b3x3_reduce, outplane_b3x3, (3, 3), (1, 1), (1, 1))),
            ('3x3_relu2', nn.ReLU(True))
        ]))

        self.c = nn.Sequential(OrderedDict([
            ('5x5_reduce', nn.Conv2d(inplane, outplane_c5x5_reduce, (1, 1), (1, 1), (0, 0))),
            ('5x5_relu1', nn.ReLU(True)),
            ('5x5', nn.Conv2d(outplane_c5x5_reduce, outplane_c5x5, (5, 5), (1, 1), (2, 2))),
            ('5x5_relu2', nn.ReLU(True))
        ]))

        self.d = nn.Sequential(OrderedDict([
            ('pool_pool', nn.MaxPool2d((3, 3), (1, 1), (1, 1))),
            ('pool_proj', nn.Conv2d(inplane, outplane_pool_proj, (1, 1), (1, 1), (0, 0))),
            ('pool_relu', nn.ReLU(True))
        ]))
        for m in self.a.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
        for m in self.b.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
        for m in self.c.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
        for m in self.d.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, input):
        return torch.cat([self.a(input), self.b(input), self.c(input), self.d(input)], 1)


class Inception(nn.Module):
    def __init__(self):
        super(Inception, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Sequential(OrderedDict([
                ('7x7_s2', nn.Conv2d(3, 64, (7, 7), (2, 2), (3, 3))),
                ('relu1', nn.ReLU(True)),
                ('pool1', nn.MaxPool2d((3, 3), (2, 2), ceil_mode=True)),
                #                 ('lrn1', nn.CrossMapLRN2d(5, 0.0001, 0.75, 1))
                ('lrn1', nn.LocalResponseNorm(5, 0.0001, 0.75, 1))
            ]))),

            ('conv2', nn.Sequential(OrderedDict([
                ('3x3_reduce', nn.Conv2d(64, 64, (1, 1), (1, 1), (0, 0))),
                ('relu1', nn.ReLU(True)),
                ('3x3', nn.Conv2d(64, 192, (3, 3), (1, 1), (1, 1))),
                ('relu2', nn.ReLU(True)),
                #                 ('lrn2', nn.CrossMapLRN2d(5, 0.0001, 0.75, 1)),
                ('lrn2', nn.LocalResponseNorm(5, 0.0001, 0.75, 1)),
                ('pool2', nn.MaxPool2d((3, 3), (2, 2), ceil_mode=True))
            ]))),

            ('inception_3a', InceptionModule(192, 64, 96, 128, 16, 32, 32)),
            ('inception_3b', InceptionModule(256, 128, 128, 192, 32, 96, 64)),

            ('pool3', nn.MaxPool2d((3, 3), (2, 2), ceil_mode=True)),

            ('inception_4a', InceptionModule(480, 192, 96, 208, 16, 48, 64)),
            ('inception_4b', InceptionModule(512, 160, 112, 224, 24, 64, 64)),
            ('inception_4c', InceptionModule(512, 128, 128, 256, 24, 64, 64)),
            ('inception_4d', InceptionModule(512, 112, 144, 288, 32, 64, 64)),
            ('inception_4e', InceptionModule(528, 256, 160, 320, 32, 128, 128)),

            ('pool4', nn.MaxPool2d((3, 3), (2, 2), ceil_mode=True)),

            ('inception_5a', InceptionModule(832, 256, 160, 320, 32, 128, 128)),
            ('inception_5b', InceptionModule(832, 384, 192, 384, 48, 128, 128)),
        ]))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-5)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)

    return output

class googlenet(nn.Module):
    def __init__(self,embedding_size, bg_embedding_size = 1024, pretrained = True, is_norm=True, is_student = True, pretrained_root=None,group_num = 4):
        super(googlenet, self).__init__()
        self.group_num = group_num
        self.model = Inception()
        if pretrained:  
            dic = {"3x3":"b", "5x5":"c", "1x1":"a", "poo":"d"}
            model_dict = self.model.state_dict()
            if pretrained_root == None:
                pretrained_dict = torch.load('./inception.pth')
            else:
                pretrained_dict = torch.load(pretrained_root)
            for key in list(pretrained_dict.keys()):
                l = key.split(".")
                if "inception" in l[0]:
                    l.insert(1, dic[l[1][:3]])
                    newkey = ".".join(l)
                else:
                    newkey = key
                newkey = "features." + newkey
                tmp = pretrained_dict[key]
                pretrained_dict[newkey] = tmp
                del pretrained_dict[key]
            pretrained_dict = {k: torch.from_numpy(v).cuda() for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
        self.is_norm = is_norm
        self.is_student = is_student
        self.embedding_size = embedding_size
        self.bg_embedding_size = bg_embedding_size
        self.num_ftrs = 1024
        self.model.gap = nn.AdaptiveAvgPool2d(1)
        self.model.gmp = nn.AdaptiveMaxPool2d(1)

        # self.model.embedding_gs = []
        # for j in range(group_num):
        #     embedding_g = nn.Linear(self.num_ftrs, self.bg_embedding_size)
        #     nn.init.orthogonal_(embedding_g.weight)
        #     nn.init.constant_(embedding_g.bias, 0)
        #     self.model.embedding_gs.append(embedding_g.cuda())

        self.model.embedding_g1 = nn.Linear(self.num_ftrs, self.bg_embedding_size)
        self.model.embedding_g2 = nn.Linear(self.num_ftrs, self.bg_embedding_size)
        self.model.embedding_g3 = nn.Linear(self.num_ftrs, self.bg_embedding_size)
        self.model.embedding_g4 = nn.Linear(self.num_ftrs, self.bg_embedding_size)
        # self.model.embedding_g5 = nn.Linear(self.num_ftrs, self.bg_embedding_size)
        nn.init.orthogonal_(self.model.embedding_g1.weight)
        nn.init.constant_(self.model.embedding_g1.bias, 0)
        nn.init.orthogonal_(self.model.embedding_g2.weight)
        nn.init.constant_(self.model.embedding_g2.bias, 0)
        nn.init.orthogonal_(self.model.embedding_g3.weight)
        nn.init.constant_(self.model.embedding_g3.bias, 0)
        nn.init.orthogonal_(self.model.embedding_g4.weight)
        nn.init.constant_(self.model.embedding_g4.bias, 0)
        # nn.init.orthogonal_(self.model.embedding_g5.weight)
        # nn.init.constant_(self.model.embedding_g5.bias, 0)

        if is_student:
            # self.model.embedding_fs = []
            # for j in range(group_num):
            #     embedding_f = nn.Linear(self.num_ftrs, self.embedding_size)
            #     nn.init.orthogonal_(embedding_f.weight)
            #     nn.init.constant_(embedding_f.bias, 0)
            #     self.model.embedding_fs.append(embedding_f.cuda())


            self.model.embedding_f1 = nn.Linear(self.num_ftrs, self.embedding_size)
            nn.init.orthogonal_(self.model.embedding_f1.weight)
            nn.init.constant_(self.model.embedding_f1.bias, 0)
            self.model.embedding_f2 = nn.Linear(self.num_ftrs, self.embedding_size)
            nn.init.orthogonal_(self.model.embedding_f2.weight)
            nn.init.constant_(self.model.embedding_f2.bias, 0)
            self.model.embedding_f3 = nn.Linear(self.num_ftrs, self.embedding_size)
            nn.init.orthogonal_(self.model.embedding_f3.weight)
            nn.init.constant_(self.model.embedding_f3.bias, 0)
            self.model.embedding_f4 = nn.Linear(self.num_ftrs, self.embedding_size)
            nn.init.orthogonal_(self.model.embedding_f4.weight)
            # nn.init.constant_(self.model.embedding_f4.bias, 0)
            # self.model.embedding_f5 = nn.Linear(self.num_ftrs, self.embedding_size)
            # nn.init.orthogonal_(self.model.embedding_f5.weight)
            # nn.init.constant_(self.model.embedding_f5.bias, 0)

    def forward(self, x, groups=None,save_feat=False):
        x = self.model.features(x)

        avg_x = self.model.gap(x)
        max_x = self.model.gmp(x)
        x = avg_x + max_x
        feat = x.view(x.size(0), -1)
        if groups == None:
            if self.is_student:
                x_f = self.model.embedding_f1(feat)
                x_g = self.model.embedding_g1(feat)
                # x_g = self.model.embedding_gs[0](logo_feat)
                # x_f = self.model.embedding_fs[0](logo_feat)
                if self.is_norm:
                    x_f = l2_norm(x_f)
                return x_g, x_f
            else:
                x_g = self.model.embedding_gs[0](feat)
                return x_g
        else:
            idxs = []
            for j in range(self.group_num):
                idx = torch.where(groups == (j+1))[0]
                idxs.append(idx)
            # idx1 = torch.where(groups == 1)[0]
            # # feat1 = logo_feat[idx1]
            # idx2 = torch.where(groups == 2)[0]
            # # feat2 = logo_feat[idx2]
            # idx3 = torch.where(groups == 3)[0]
            # # feat3 = logo_feat[idx3]

            if save_feat == True:
                return feat

            if self.is_student:
                # x_fs = []
                # for j in range(self.group_num):
                #     x_f = self.model.embedding_fs[j](logo_feat)
                #     if self.is_norm:
                #         x_f = l2_norm(x_f)
                #     x_fs.append(x_f)

                x_f1 = self.model.embedding_f1(feat)
                x_f2 = self.model.embedding_f2(feat)
                x_f3 = self.model.embedding_f3(feat)
                x_f4 = self.model.embedding_f4(feat)
                # x_f5 = self.model.embedding_f5(logo_feat)
                if self.is_norm:
                    x_f1 = l2_norm(x_f1)
                    x_f2 = l2_norm(x_f2)
                    x_f3 = l2_norm(x_f3)
                    x_f4 = l2_norm(x_f4)
                    # x_f5 = l2_norm(x_f5)

            # x_gs = []
            # for j in range(self.group_num):
            #     x_g = self.model.embedding_gs[j](logo_feat)
            #     x_gs.append(x_g)

            x_g1 = self.model.embedding_g1(feat)
            x_g2 = self.model.embedding_g2(feat)
            x_g3 = self.model.embedding_g3(feat)
            x_g4 = self.model.embedding_g4(feat)
            # x_g5 = self.model.embedding_g5(logo_feat)

            if self.is_student:
                return [x_g1, x_g2,x_g3,x_g4], [x_f1,x_f2,x_f3,x_f4],idxs
            else:
                return [x_g1, x_g2,x_g3,x_g4], idxs
                # return [x_g1, x_g2, x_g3],[idx1,idx2,idx3]




