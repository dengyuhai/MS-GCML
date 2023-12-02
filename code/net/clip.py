import torch
import torch.nn as nn
import warnings
from typing import Union,Tuple
from collections import OrderedDict
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import numpy as np
from PIL import Image
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-5)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)

    return output
def _convert_image_to_rgb(image):
    return image.convert("RGB")
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length

        vision_heads = vision_width // 64
        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim
        )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text

def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    # convert_weights(model)
    model.load_state_dict(state_dict)
    # return model.eval()
    return model

def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
         jit: bool = False, pretrained_root: str = None):
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict

    device : Union[str, torch.device]
        The device to put the loaded model

    jit : bool
        Whether to load the optimized JIT model or more hackable non-JIT model (default).

    download_root: str
        path to download the model files; by default, it uses "~/.cache/clip"

    Returns
    -------
    model : torch.nn.Module
        The CLIP model

    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    if pretrained_root == None:
        model_path = '/home/msi/PycharmProjects/CLIP/ViT-B-16.pt'
    else:
        model_path = pretrained_root

    with open(model_path, 'rb') as opened_file:
        try:
            # loading JIT archive
            model = torch.jit.load(opened_file, map_location=device if jit else "cpu").eval()
            state_dict = None
        except RuntimeError:
            # loading saved state dict
            if jit:
                warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
                jit = False
            state_dict = torch.load(opened_file, map_location="cpu")

    model = build_model(state_dict or model.state_dict()).to(device)
    if str(device) == "cpu":
        model.float()
    return model, _transform(model.visual.input_resolution)

class clip_image_encoder(nn.Module):
    def __init__(self,full_clip):
        super(clip_image_encoder, self).__init__()
        self.encoder = full_clip.visual
        self.dtype = full_clip.dtype

    def forward(self, image):
        return self.encoder(image.type(self.dtype))


class clip_gcl(nn.Module):
    def __init__(self,embedding_size, bg_embedding_size = 1024, pretrained = True, is_norm=True, is_student = True, pretrained_root=None,group_num = 4):
        super(clip_gcl, self).__init__()
        self.group_num = group_num
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if pretrained:
            if pretrained_root == None:
                pretrained_weights = '/home/msi/PycharmProjects/CLIP/ViT-B-16.pt'
            else:
                pretrained_weights = pretrained_root
            model, preprocess = load('ViT-B/16', device,pretrained_root=pretrained_weights)
            self.preprocess = preprocess
        self.device = device
        self.model = clip_image_encoder(model)

        self.is_norm = is_norm
        self.is_student = is_student
        self.embedding_size = embedding_size
        self.bg_embedding_size = bg_embedding_size
        self.num_ftrs = 512
        # self.model.gap = nn.AdaptiveAvgPool2d(1)
        # self.model.gmp = nn.AdaptiveMaxPool2d(1)

        self.model.embedding_g1 = nn.Linear(self.num_ftrs, self.bg_embedding_size)
        self.model.embedding_g2 = nn.Linear(self.num_ftrs, self.bg_embedding_size)
        self.model.embedding_g3 = nn.Linear(self.num_ftrs, self.bg_embedding_size)
        self.model.embedding_g4 = nn.Linear(self.num_ftrs, self.bg_embedding_size)

        nn.init.xavier_uniform_(self.model.embedding_g1.weight)
        nn.init.constant_(self.model.embedding_g1.bias, 0)
        nn.init.xavier_uniform_(self.model.embedding_g2.weight)
        nn.init.constant_(self.model.embedding_g2.bias, 0)
        nn.init.xavier_uniform_(self.model.embedding_g3.weight)
        nn.init.constant_(self.model.embedding_g3.bias, 0)
        nn.init.xavier_uniform_(self.model.embedding_g4.weight)
        nn.init.constant_(self.model.embedding_g4.bias, 0)

        if is_student:
            # self.model.embedding_fs = []
            # for j in range(group_num):
            #     embedding_f = nn.Linear(self.num_ftrs, self.embedding_size)
            #     nn.init.xavier_uniform_(embedding_f.weight)
            #     nn.init.constant_(embedding_f.bias, 0)
            #     self.model.embedding_fs.append(embedding_f.cuda())

            self.model.embedding_f1 = nn.Linear(self.num_ftrs, self.embedding_size)
            nn.init.xavier_uniform_(self.model.embedding_f1.weight)
            nn.init.constant_(self.model.embedding_f1.bias, 0)
            self.model.embedding_f2 = nn.Linear(self.num_ftrs, self.embedding_size)
            nn.init.xavier_uniform_(self.model.embedding_f2.weight)
            nn.init.constant_(self.model.embedding_f2.bias, 0)
            self.model.embedding_f3 = nn.Linear(self.num_ftrs, self.embedding_size)
            nn.init.xavier_uniform_(self.model.embedding_f3.weight)
            nn.init.constant_(self.model.embedding_f3.bias, 0)
            self.model.embedding_f4 = nn.Linear(self.num_ftrs, self.embedding_size)
            nn.init.xavier_uniform_(self.model.embedding_f4.weight)

    def forward(self, x, groups=None, save_feat=False):
        feat = self.model.encoder(x)
        if groups == None:
            if self.is_student:
                # x_f = feat
                
                x_g = self.model.embedding_g1(feat)
                x_f = self.model.embedding_f1(feat)
                x_f = l2_norm(x_f)
                return x_g, x_f
            else:
                x_g = self.model.embedding_gxx(feat)
                return x_g
        else:
            idxs = []
            for j in range(self.group_num):
                idx = torch.where(groups == (j + 1))[0]
                idxs.append(idx)

            if save_feat == True:
                return feat
            # group1_idx = idxs[0]#.cuda()
            # group2_idx = torch.cat([idxs[0],idxs[1]])#.cuda()
            # group3_idx = torch.cat([idxs[0], idxs[1],idxs[2]])#.cuda()
            # group4_idx = torch.cat([idxs[0], idxs[1], idxs[2],idxs[3]])#.cuda()
            if self.is_student:
                # x_f1 = self.model.embedding_f1(logo_feat[group1_idx,:])
                # x_f2 = self.model.embedding_f2(logo_feat[group2_idx,:])
                # x_f3 = self.model.embedding_f3(logo_feat[group3_idx,:])
                # x_f4 = self.model.embedding_f4(logo_feat[group4_idx,:])
                x_f1 = self.model.embedding_f1(feat)
                x_f2 = self.model.embedding_f2(feat)
                x_f3 = self.model.embedding_f3(feat)
                x_f4 = self.model.embedding_f4(feat)
                # if self.is_norm:
                #     x_f1 = l2_norm(x_f1)
                #     x_f2 = l2_norm(x_f2)
                #     x_f3 = l2_norm(x_f3)
                #     x_f4 = l2_norm(x_f4)
                x_f1 = l2_norm(x_f1)
                x_f2 = l2_norm(x_f2)
                x_f3 = l2_norm(x_f3)
                x_f4 = l2_norm(x_f4)


            # x_g1 = self.model.embedding_g1(logo_feat[group1_idx,:])
            # x_g2 = self.model.embedding_g2(logo_feat[group2_idx,:])
            # x_g3 = self.model.embedding_g3(logo_feat[group3_idx,:])
            # x_g4 = self.model.embedding_g4(logo_feat[group4_idx,:])
            x_g1 = self.model.embedding_g1(feat)
            x_g2 = self.model.embedding_g2(feat)
            x_g3 = self.model.embedding_g3(feat)
            x_g4 = self.model.embedding_g4(feat)
            del feat#,group1_idx,group2_idx,group3_idx,group4_idx
            if self.is_student:
                return [x_g1, x_g2, x_g3, x_g4], [x_f1, x_f2, x_f3, x_f4], idxs
            else:
                return [x_g1, x_g2, x_g3, x_g4], idxs
