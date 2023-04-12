from itertools import repeat
import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = tuple(repeat(img_size, 2))
        patch_size = tuple(repeat(patch_size, 2))
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.W = 0
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        dim = dim // 2
        self.norm1 = norm_layer(dim)
        self.time_norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # self.time_attn = TimeAttention(self.attn.qkv, dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.time_attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.time_norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.time_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.linear = nn.Linear(dim * 2, dim * 2)

    def forward(self, x):
        N, L, T, D = x.shape

        space_x, time_x = x.split([D // 2, D // 2], dim=-1)
        space_x = space_x.reshape(shape=(N * T, L, D // 2))
        time_x = time_x.reshape(shape=(N * L, T, D // 2))

        space_x = self.attn(self.norm1(space_x))
        space_x = space_x + self.drop_path(space_x)
        space_x = space_x + self.drop_path(self.mlp(self.norm2(space_x)))

        time_x = time_x + self.time_attn(self.time_norm1(time_x))
        time_x = time_x + self.drop_path(time_x)
        time_x = time_x + self.drop_path(self.time_mlp(self.time_norm2(time_x)))

        space_x = space_x.reshape(shape=(N, L, T, D // 2))
        time_x = time_x.reshape(shape=(N, L, T, D // 2))

        x = torch.concat((space_x ,time_x), dim=-1)
        x = self.linear(x)

        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        # space qkv
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # space q * k
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # space q * k * v
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class TimeAttention(nn.Module):
    def __init__(self, qkv, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = qkv
        self.time_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.time_proj = nn.Linear(dim, dim)

    def forward(self, x, y):
        B, N, C = x.shape
        T, _, _, _ = y.shape

        # space qkv
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # time qkv
        time_qkv = self.time_qkv(y).reshape(T, B, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        time_q, time_k, time_v = time_qkv.unbind(0)   #(T, B, num_heads, N, c//num_heads)

        # time q * k
        tk = time_k[0]
        for i in range(1, T):
            tk = tk * time_k[i]
        time_attn = (q * tk) * self.scale
        time_attn = time_attn.softmax(dim=-1)
        time_attn = self.attn_drop(time_attn)

        # time q * k * v
        tv = time_v[0]
        for i in range(1, T):
            tv = tv * time_v[i]
        y = (time_attn * tv).transpose(1, 2).reshape(B, N, C)

        y = self.time_proj(y)
        y = self.proj_drop(y)

        return y


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = tuple(repeat(drop, 2))

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):

    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):

    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
