import random
import time
from functools import partial
from itertools import repeat
import torch
import torch.nn as nn
import numpy as np
from timm.models.vision_transformer import PatchEmbed, Block
import math
from util.pos_embed import get_2d_sincos_pos_embed
from util.metrics import loss_c

class FViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=32, patch_size=2, in_chans=1, sample_index=None, fmargin=20,
                 embed_dim=1024, depth=24, num_heads=16, random=random, type='softmax',
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, mask_ratio=0.9375):
        super().__init__()
        self.poi = np.load("data/POI_32.npy")

        self.fmargin = fmargin
        self.type = type
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.mask_ratio = mask_ratio
        self.r = int(1 / (1 - self.mask_ratio)) # 16
        self.random_flag = random

        self.sample_index = sample_index

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.s_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.pos_embed2 = nn.Parameter(torch.zeros(1, int(num_patches / self.r) + 1, embed_dim), requires_grad=False)
        self.pos_embed3 = nn.Parameter(torch.zeros(1, int(num_patches / self.r) + 1, embed_dim), requires_grad=False) # sapce

        self.in_chans = in_chans
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # M2-Normalization
        r = int(math.sqrt(1 / (1 - self.mask_ratio)))
        self.den_softmax = N2_Normalization(r)
        self.recover = Recover_from_density(r)


        self.norm_pix_loss = norm_pix_loss

        self.fpoi_embed = nn.Linear(14, 1, bias=True)
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * self.in_chans))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, h * p))
        return imgs

    def Start_order(self, n, H, d):
        # n_d个起始位置下标
        n_d = int(math.sqrt(n))
        start = [0 for i in range(n)]

        for i in range(n_d):
            for j in range(n_d):
                start[i * n_d + j] = i * H * d + j * d
        return start

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, l, D = x.shape  # batch, length, dim
        r = int(1 / (1.0 - mask_ratio)) # 16
        L = l * r
        self.L = L
        nl = [i % L for i in range(N * L)]
        data = np.array(nl).reshape(N, L)

        n = int(L / r)
        H = int(math.sqrt(L))
        d = int(math.sqrt(r))

        if self.random_flag == 'fixed':
            start_order = self.Start_order(n, H, d)

            for oid, i in enumerate(start_order):
                data[:, i] = -data[:, i]

            # print(data[0])

        elif self.random_flag == 'random':
            # 0-r序列，从中挑选
            t = [0 for i in range(r)]
            start_order = self.Start_order(n, H, d)
            for p in range(N):
                for i in start_order:
                    for j in range(d):
                        for k in range(d):
                            t[j * d + k] = i + j * H + k
                    save_id = random.choice(t)
                    data[p][save_id] = save_id * -1
        else :
            print("Invalid random value!!", random)

        noise = torch.tensor(data).cuda()

        len_keep = l
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        ids_keep, _ = torch.sort(ids_keep, 1)
        ids_shuffle = torch.cat((ids_keep, ids_shuffle[:, len_keep:]), 1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        self.mask = mask

        self.space_roate()

        self.pos_embed2[:, :1, :] = self.pos_embed[:, :1, :]
        self.pos_embed2[:, 1:, :] = torch.gather(self.pos_embed.data, dim=1,
                                               index=ids_keep.add(1).unsqueeze(-1).repeat(1, 1, x.shape[2])[0].unsqueeze(0))

        self.pos_embed3[:, :1, :] = self.pos_embed[:, :1, :]
        self.pos_embed3[:, 1:, :] = torch.gather(self.pos_embed.data, dim=1,
                                                 index=self.id_s_r_er_keep.add(1).unsqueeze(-1).unsqueeze(0).repeat(1, 1, x.shape[2]))

        return ids_restore

    def forward_encoder(self, x, s_x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)
        s_x = self.patch_embed(s_x)

        # masking: length -> length * mask_ratio
        ids_restore = self.random_masking(x, mask_ratio)

        # add pos embed w/o cls token
        x = x + self.pos_embed2[:, 1:, :]
        s_x = s_x + self.pos_embed3[:, 1:, :]

        # append cls token
        # cls_token = self.cls_token + self.pos_embed2[:, :1, :]
        # cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        # x = torch.cat((cls_tokens, x), dim=1)
        #
        # s_cls_token = self.s_cls_token + self.pos_embed3[:, :1, :]
        # s_cls_tokens = s_cls_token.expand(s_x.shape[0], -1, -1)
        # s_x = torch.cat((s_cls_tokens, s_x), dim=1)


        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
            s_x = blk(x)

        x = self.norm(x)
        s_x = self.norm(s_x)

        return x, s_x, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        # x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed


        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        # x = x[:, 1:, :]

        return x

    # Encoder Space Loss
    def forward_sloss(self, s_latent, s_r_latent):
        id_index = self.id_s_r_er.unsqueeze(0).unsqueeze(-1).repeat(s_r_latent.shape[0], 1, s_r_latent.shape[2])
        S_R_latent = torch.gather(s_r_latent, dim=1, index=id_index)

        s_loss = (s_latent - S_R_latent) ** 2
        s_loss = s_loss.mean()  # [N, L], mean loss per patch

        return s_loss

    # Decoder Space Loss
    def forward_sdloss(self, s_pred, s_r_pred):
        id_index = self.id_s_r_er_all.unsqueeze(0).unsqueeze(-1).repeat(s_r_pred.shape[0], 1, s_r_pred.shape[2])
        S_R_pred = torch.gather(s_r_pred, dim=1, index=id_index)

        sd_loss = (s_pred - S_R_pred) ** 2
        sd_loss = sd_loss.mean()  # [N, L], mean loss per patch

        return sd_loss

    # Decoder POI loss
    def forward_poiloss(self, latent):
        poi = self.poi
        poi = torch.from_numpy(poi).cuda().float()
        poi = self.fpoi_embed(poi).squeeze(-1)
        loss = loss_c(latent, poi, self.fmargin, self.type)

        return loss

    def forward_loss(self, target, pred):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        criteria = nn.MSELoss()
        loss = criteria(pred, target)

        return loss

    # 处理旋转后的Map的 丢失Patch索引和 保存Patch索引
    def space_roate(self):
        LL = self.L  # 64
        l = int(LL / self.r)
        n = int(math.sqrt(LL))
        s_er = np.array([i for i in range(LL)])
        s_er = torch.from_numpy(s_er).cuda()

        masked = self.mask[0] * -1000000
        s_er = s_er + masked
        s_er = s_er.reshape(n, n)
        s_r_er = torch.rot90(s_er, self.k, [0, 1])

        # 对应旋转前的索引
        id_s_r_er = torch.zeros([l])
        id_s_r_er_all = torch.zeros([LL])

        # 旋转后的keep，drop 索引
        id_s_r_er_keep = torch.zeros([l])
        id_s_r_er_drop = torch.zeros([LL - l])

        k, p = 0, 0
        for i in range(n):
            for j in range(n):
                if s_r_er[i][j] < 0:
                    id_s_r_er_all[i * n + j] = s_r_er[i][j] + 1000000
                    id_s_r_er_drop[p] = i * n + j
                    p = p + 1
                else:
                    id_s_r_er_all[i * n + j] = s_r_er[i][j]
                    id_s_r_er[k] = s_r_er[i][j]
                    id_s_r_er_keep[k] = i * n + j
                    k = k + 1

        t, ids_shuffle = torch.sort(id_s_r_er, dim=0)
        for i in range(len(t)):
            t[i] = i
        id_restore = torch.argsort(ids_shuffle, dim=0)
        id_s_r_er = torch.gather(t, dim=0, index=id_restore)

        id_s_r_er_shuffle = torch.cat((id_s_r_er_drop, id_s_r_er_keep), dim=0)

        self.id_s_r_er_restore = torch.argsort(id_s_r_er_shuffle, dim=0).type(torch.int64).cuda()
        self.id_s_r_er_keep = id_s_r_er_keep.type(torch.int64).cuda()

        self.id_s_r_er = id_s_r_er.type(torch.int64).cuda()
        self.id_s_r_er_all = id_s_r_er_all.type(torch.int64).cuda()


    def forward(self, imgs, target, mask_ratio=0.75):
        # N, C, H, W = HX.shape
        self.k = random.randint(0, 3)
        s_x = torch.rot90(imgs, self.k, [2, 3])

        latent, space_latent, ids_restore = self.forward_encoder(imgs, s_x, mask_ratio)
        s_loss = self.forward_sloss(latent, space_latent)

        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]

        # get the distribution matrix
        # print(target.shape)
        N, L, D = pred.shape
        n = int(math.sqrt(L * D))
        pred = pred.reshape(-1, n, n)

        xlatent = pred.reshape(-1, L * D, 1)
        poi_loss = self.forward_poiloss(xlatent)

        dis = self.den_softmax(pred)

        # recover fine-grained flows from coarse-grained flows and distributions
        pred = self.recover(dis, imgs)
        loss = self.forward_loss(target, pred)
        return loss, poi_loss, s_loss, pred



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

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):

        B, C, H, W = x.shape

        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class N2_Normalization(nn.Module):
    def __init__(self, upscale_factor):
        super(N2_Normalization, self).__init__()
        self.upscale_factor = upscale_factor
        self.avgpool = nn.AvgPool2d(upscale_factor)
        self.upsample = nn.Upsample(
            scale_factor=upscale_factor, mode='nearest')
        self.epsilon = 1e-5

    def forward(self, x):
        out = self.avgpool(x) * self.upscale_factor ** 2 # sum pooling

        out = out.unsqueeze(1)
        out = self.upsample(out)
        out = out.squeeze(1)

        return torch.div(x, out + self.epsilon)

class Recover_from_density(nn.Module):
    def __init__(self, upscale_factor):
        super(Recover_from_density, self).__init__()
        self.upscale_factor = upscale_factor
        self.upsample = nn.Upsample(
            scale_factor=upscale_factor, mode='nearest')

    def forward(self, x, lr_img):
        out = self.upsample(lr_img)
        x = x.unsqueeze(1)
        return torch.mul(x, out)


def mae_vit_1(patch_size, in_chans, **kwargs):
    model = FViT(
        patch_size=patch_size, embed_dim=64, depth=4, num_heads=4, in_chans=in_chans,
        decoder_embed_dim=32, decoder_depth=2, decoder_num_heads=4,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_2(patch_size, in_chans, **kwargs):
    model = FViT(
        patch_size=patch_size, embed_dim=64, depth=6, num_heads=8, in_chans=in_chans,
        decoder_embed_dim=32, decoder_depth=4, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_3(patch_size, in_chans, **kwargs):
    model = FViT(
        patch_size=patch_size, embed_dim=128, depth=4, num_heads=4, in_chans=in_chans,
        decoder_embed_dim=64, decoder_depth=2, decoder_num_heads=4,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_4(patch_size, in_chans, **kwargs):
    model = FViT(
        patch_size=patch_size, embed_dim=256, depth=10, num_heads=8, in_chans=in_chans,
        decoder_embed_dim=128, decoder_depth=5, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_5(patch_size, in_chans, **kwargs):
    model = FViT(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, in_chans=in_chans,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
