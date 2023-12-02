import torch
import torch.nn as nn
from functools import partial, reduce
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.layers import PatchEmbed
import math
from operator import mul
import torchvision.transforms as transforms

class VisionTransformerMoCo(VisionTransformer):
    def __init__(self, stop_grad_conv1=False, **kwargs):
        super().__init__(**kwargs)
        # Use fixed 2D sin-cos position embedding
        self.build_2d_sincos_position_embedding()

        # weight initialization
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if 'qkv' in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                else:
                    nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.cls_token, std=1e-6)

        if isinstance(self.patch_embed, PatchEmbed):
            # xavier_uniform initialization
            val = math.sqrt(6. / float(3 * reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim))
            nn.init.uniform_(self.patch_embed.proj.weight, -val, val)
            nn.init.zeros_(self.patch_embed.proj.bias)

            if stop_grad_conv1:
                self.patch_embed.proj.weight.requires_grad = False
                self.patch_embed.proj.bias.requires_grad = False

    def build_2d_sincos_position_embedding(self, temperature=10000.):
        h, w = self.patch_embed.grid_size
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert self.embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = self.embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature**omega)
        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

        assert self.num_tokens == 1, 'Assuming one and only one token, [cls]'
        pe_token = torch.zeros([1, 1, self.embed_dim], dtype=torch.float32)
        self.pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
        self.pos_embed.requires_grad = False

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-5)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)

    return output


def vit_base(**kwargs):
    model = VisionTransformerMoCo(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

class moco_gcl(nn.Module):
    def __init__(self,embedding_size, bg_embedding_size = 1024, pretrained = True, is_norm=True, is_student = True, pretrained_root=None,group_num = 4):
        super(moco_gcl, self).__init__()
        self.group_num = group_num
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        _transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        self.preprocess = _transform
        model = vit_base()
        linear_keyword = 'head'
        if pretrained:
            if pretrained_root == None:
                pretrained_weights = '/home/msi/PycharmProjects/moco/vit-b-300ep.pth.zip'
            else:
                pretrained_weights = pretrained_root

            checkpoint = torch.load(pretrained_weights, map_location="cpu")
            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only base_encoder up to before the embedding layer
                if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
                    # remove prefix
                    state_dict[k[len("module.base_encoder."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]
            msg = model.load_state_dict(state_dict, strict=False)
            assert set(msg.missing_keys) == {"%s.weight" % linear_keyword, "%s.bias" % linear_keyword}
            self.model = model
        self.is_norm = is_norm
        self.is_student = is_student
        self.embedding_size = embedding_size
        self.bg_embedding_size = bg_embedding_size
        self.num_ftrs = 1000
        self.model.gap = nn.AdaptiveAvgPool2d(1)
        self.model.gmp = nn.AdaptiveMaxPool2d(1)

        self.model.embedding_g1 = nn.Linear(self.num_ftrs, self.bg_embedding_size)
        self.model.embedding_g2 = nn.Linear(self.num_ftrs, self.bg_embedding_size)
        self.model.embedding_g3 = nn.Linear(self.num_ftrs, self.bg_embedding_size)
        self.model.embedding_g4 = nn.Linear(self.num_ftrs, self.bg_embedding_size)

        nn.init.orthogonal_(self.model.embedding_g1.weight)
        nn.init.orthogonal_(self.model.embedding_g2.weight)
        nn.init.orthogonal_(self.model.embedding_g3.weight)
        nn.init.orthogonal_(self.model.embedding_g4.weight)
        nn.init.constant_(self.model.embedding_g1.bias, 0)
        nn.init.constant_(self.model.embedding_g2.bias, 0)
        nn.init.constant_(self.model.embedding_g3.bias, 0)
        nn.init.constant_(self.model.embedding_g4.bias, 0)
        if is_student:
            self.model.embedding_f1 = nn.Linear(self.num_ftrs, self.embedding_size)
            self.model.embedding_f2 = nn.Linear(self.num_ftrs, self.embedding_size)
            self.model.embedding_f3 = nn.Linear(self.num_ftrs, self.embedding_size)
            self.model.embedding_f4 = nn.Linear(self.num_ftrs, self.embedding_size)
            nn.init.constant_(self.model.embedding_f1.bias, 0)
            nn.init.constant_(self.model.embedding_f2.bias, 0)
            nn.init.constant_(self.model.embedding_f3.bias, 0)
            nn.init.constant_(self.model.embedding_f4.bias, 0)

            nn.init.orthogonal_(self.model.embedding_f1.weight)
            nn.init.orthogonal_(self.model.embedding_f2.weight)
            nn.init.orthogonal_(self.model.embedding_f3.weight)
            nn.init.orthogonal_(self.model.embedding_f4.weight)

    def forward(self, x, groups=None,save_feat=False):
        x = self.model(x)
        # x = multi_scale(x, self.model)

        # avg_x = self.model.gap(x)
        # max_x = self.model.gmp(x)
        # x = avg_x + max_x
        feat = x
        if groups == None:
            if self.is_student:
                # x_f = x
                x_g = self.model.embedding_g1(feat)
                
                x_f = self.model.embedding_f1(feat)
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

            if save_feat == True:
                return feat

            if self.is_student:
                x_f1 = self.model.embedding_f1(feat)
                x_f2 = self.model.embedding_f2(feat)
                x_f3 = self.model.embedding_f3(feat)
                x_f4 = self.model.embedding_f4(feat)

                x_f1 = l2_norm(x_f1)
                x_f2 = l2_norm(x_f2)
                x_f3 = l2_norm(x_f3)
                x_f4 = l2_norm(x_f4)

            x_g1 = self.model.embedding_g1(feat)
            x_g2 = self.model.embedding_g2(feat)
            x_g3 = self.model.embedding_g3(feat)
            x_g4 = self.model.embedding_g4(feat)

            if self.is_student:
                return [x_g1, x_g2,x_g3,x_g4], [x_f1,x_f2,x_f3,x_f4],idxs
            else:
                return [x_g1, x_g2,x_g3,x_g4], idxs

