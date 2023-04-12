import random
import time
from functools import partial
from itertools import repeat
import torch
import torch.nn as nn
import numpy as np
import ST_transformer
from timm.models.vision_transformer import Block
import math
from util.pos_embed import get_2d_sincos_pos_embed
import TimeSformer
from einops import rearrange, reduce, repeat

# fix the seed for reproducibility
seed = 2022
torch.manual_seed(seed)
random.seed(seed)


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=8, patch_size=1, in_chans=1, sample_index=None,
                 embed_dim=1024, depth=24, num_heads=16, fraction=20, T_len=5,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = ST_transformer.PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.fraction = fraction
        self.sample_index = sample_index
        # 旋转90，增加一个Map
        self.T_len = T_len + 1
        self.attention_type = 'divided_space_time'

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.time_cls_token = nn.Parameter(torch.zeros(self.T_len, 1, 1, embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding
        dpr = [x.item() for x in torch.linspace(0, 0.1, depth)]  # stochastic depth decay rule
        self.time_blocks = nn.ModuleList([
            ST_transformer.Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True,
                drop=0., attn_drop=0., drop_path=dpr[i], norm_layer=norm_layer)
            # TimeSformer.Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, qk_scale=None,
            #     drop=0., attn_drop=0., drop_path=dpr[i], norm_layer=norm_layer, attention_type=self.attention_type)
            for i in range(depth)])

        self.time_embed = nn.Parameter(torch.zeros(1, self.T_len + 1, embed_dim))
        # self.time_embed = nn.Parameter(torch.zeros(1, self.T_len, embed_dim))

        self.time_drop = nn.Dropout(p=0.)
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # (N, L ,D)
        self.time_mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim)) # (N,L,T,D)

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        # self.decoder_blocks = nn.ModuleList([
        #     Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
        #     for i in range(decoder_depth)])

        # self.time_decoder_blocks = nn.ModuleList([
        #     ST_transformer.Block(dim=decoder_embed_dim, num_heads=decoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True,
        #                          drop=0., attn_drop=0., drop_path=dpr[i], norm_layer=norm_layer)
        #     # TimeSformer.Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, qk_scale=None,
        #     #     drop=0., attn_drop=0., drop_path=dpr[i], norm_layer=norm_layer, attention_type=self.attention_type)
        #     for i in range(decoder_depth)])

        self.time_decoder_embed = nn.Parameter(torch.zeros(1, self.T_len + 1, embed_dim))
        self.time_decoder_drop = nn.Dropout(p=0.)


        self.decoder_norm = norm_layer(embed_dim)
        self.decoder_pred = nn.Linear(embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches ** .5), cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        # torch.nn.init.normal_(self.cls_token, std=.02)
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
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 1))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, hx, x):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        nl = [i % L for i in range(N * L)]
        data = np.array(nl).reshape(N, L)

        for i in self.sample_index:
            data[:, i] = -data[:, i]

        noise = torch.tensor(data).cuda()

        len_drop = len(self.sample_index)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, len_drop:]
        ids_drop = ids_shuffle[:, :len_drop]
        ids_drop, _ = torch.sort(ids_drop, 1)
        ids_shuffle = torch.cat((ids_drop, ids_keep), 1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        mask = torch.ones([N, L], device=x.device)
        mask[:, len_drop:] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        self.mask = mask

        self.space_roate()

        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        hx_masked = []
        for i in range(hx.shape[0]):
            # 处理 旋转后的Map的 保留Patch， Mask，Encoder部分 暂时不变
            if i == 0:
                hx_masked.append(torch.gather(hx[i], dim=1, index=self.id_s_r_er_keep.unsqueeze(0).unsqueeze(-1).repeat(hx[i].shape[0], 1, D)))
            else:
                hx_masked.append(torch.gather(hx[i], dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)))
        hx_masked = torch.stack(hx_masked, dim=0)

        return x_masked, hx_masked, mask, ids_restore

    def forward_encoder(self, hx, x):

        Phx = []
        # embed patches
        x = self.patch_embed(x)
        for i in range(hx.shape[0]):
            phx = self.patch_embed(hx[i])
            phx = phx + self.pos_embed[:, 1:, :]
            Phx.append(phx)

        HX = torch.stack(Phx, dim=0)
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]


        # masking: length -> length * mask_ratio
        x, hx, mask, ids_restore = self.random_masking(HX, x)


        # TimeSformer
        x = x.unsqueeze(0)
        x = torch.cat((x, hx), dim=0)
        # append time embed
        T, B, L, D = x.shape
        x = rearrange(x, 't b l d -> (b l) t d')
        x = x + self.time_embed
        x = self.time_drop(x)
        x = rearrange(x, '(b l) t d -> b l t d', b=B, t=T)
        # append cls token
        # cls_token = self.cls_token + self.pos_embed[:, :1, :]
        # cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        # x = torch.cat((cls_tokens, x), dim=1)


        # ST_transformer
        # time_cls_token = self.time_cls_token + self.pos_embed[:, :1, :].unsqueeze(0).repeat(self.T_len, 1, 1, 1)
        # time_cls_token = time_cls_token.expand(-1, hx.shape[1], -1, -1)
        # hx = torch.cat((time_cls_token, hx), dim=2)
        # T, B, L, D = hx.shape
        # hx = rearrange(hx, 't b l d -> (b l) t d')
        # hx = hx + self.time_embed
        # hx = self.time_drop(hx)
        # hx = rearrange(hx, '(b l) t d -> t b l d', b=B, t=T)
        # cls_token = self.cls_token + self.pos_embed[:, :1, :]
        # cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        # x = torch.cat((cls_tokens, x), dim=1)


        # apply Space_Time-Transformer blocks
        for blk in self.time_blocks:
            x = blk(x)
            # x = blk(x, B, T)

        # space loss
        s_x = x[:, :, 0, :]
        s_r_x = x[:, :, 1, :]

        # time loss
        start = 2 + self.clen
        time_x = x[:, :, start : start + self.plen, :]

        x = self.norm(x)

        # TimeSformer
        # tokens = x[:, :1, :]
        # x = rearrange(x[:, 1:, :], 'b (l t) d -> t b l d', b=B, l=L)
        # x = torch.cat((tokens, x[0]), dim=1)

        return x, s_x, s_r_x, time_x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        # mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        time_mask_tokens = self.time_mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], x.shape[2], 1)

        # x_ = torch.cat([x[:, :, :], mask_tokens], dim=1)  # no cls token
        # x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        time_x_ = torch.cat([x[:, :, :, :], time_mask_tokens], dim=1)

        time_x = []
        for i in range(time_x_.shape[2]):
            if i == 1:
                time_x.append(torch.gather(time_x_[:, :, i, :], dim=1,
                                       index=self.id_s_r_er_restore.unsqueeze(0).unsqueeze(-1).repeat(x.shape[0], 1,
                                                                                                      x.shape[3])))
            else:
                time_x.append(torch.gather(time_x_[:, :, i, :], dim=1,
                         index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[3])))
        x = torch.stack(time_x, dim=2)

        # add pos embed
        # x = x + self.decoder_pos_embed

        B, L, T, D = x.shape
        x = x + self.decoder_pos_embed.unsqueeze(-2).repeat(1, 1, x.shape[2], 1)
        x = rearrange(x, 'b l t d -> (b l) t d')
        x = x + self.time_decoder_embed
        x = self.time_decoder_drop(x)
        x = rearrange(x, '(b l) t d -> b l t d', b=B, l=L)


        # apply Transformer blocks
        for blk in self.time_blocks:
            x = blk(x)

        # space loss
        s_x = x[:, :, 0, :]
        s_r_x = x[:, :, 1, :]

        # time loss
        start = 2 + self.clen
        time_x = x[:, :, start: start + self.plen, :]

        x = x[:, :, 0, :]

        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        # x = x[:, 1:, :]

        return x, s_x, s_r_x, time_x


    def forward_loss(self, target, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """

        target = self.patchify(target)
        # if self.norm_pix_loss:
        #     mean = target.mean(dim=-1, keepdim=True)
        #     var = target.var(dim=-1, keepdim=True)
        #     target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        # 所有块 计算 mse loss
        all_loss = loss.mean()

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss, all_loss

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

    # Time Loss
    def forward_tloss(self, s_latent, time_latent):
        time_latent = torch.mean(time_latent, dim=2)
        loss = (s_latent - time_latent) ** 2
        loss = loss.mean()

        return loss

    # 处理旋转后的Map的 丢失Patch索引和 保存Patch索引
    def space_roate(self):
        LL = self.patch_embed.num_patches
        n = int(math.sqrt(LL))
        s_er = np.array([i for i in range(LL)])
        s_er = torch.from_numpy(s_er).cuda()

        masked = self.mask[0] * -1000000
        s_er = s_er + masked
        s_er = s_er.reshape(n, n)
        s_r_er = torch.rot90(s_er, self.k, [0, 1])

        # 对应旋转前的索引
        id_s_r_er = torch.zeros([LL - len(self.sample_index)])
        id_s_r_er_all = torch.zeros([LL])

        # 旋转后的keep，drop 索引
        id_s_r_er_keep = torch.zeros([LL - len(self.sample_index)])
        id_s_r_er_drop = torch.zeros([len(self.sample_index)])

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

    def forward(self, *args):
        if len(args) == 3:
            XC, X, target = args
            HX = XC.permute(1, 0, 2, 3, 4)
            self.clen = XC.shape[1]
            self.plen = 0
            self.tlen = 0

        elif len(args) == 4:
            XC, XP, X, target = args
            HX = torch.cat((XC, XP), 1).permute(1, 0, 2, 3, 4)
            self.clen = XC.shape[1]
            self.plen = XP.shape[1]
            self.tlen = 0

        else:
            XC, XP, XT, X, target = args
            HX = torch.cat((XC, XP, XT), 1).permute(1, 0, 2, 3, 4)
            self.clen = XC.shape[1]
            self.plen = XP.shape[1]
            self.tlen = XT.shape[1]


        # T, N, C, H, W = HX.shape
        self.k = random.randint(0, 3)
        S_X = torch.rot90(X, self.k, [2, 3]).unsqueeze(0)
        HX = torch.cat((S_X, HX), 0)

        latent, s_latent, s_r_latent, time_latent, mask, ids_restore = self.forward_encoder(HX, X)

        s_loss = self.forward_sloss(s_latent, s_r_latent)
        t_loss = self.forward_tloss(s_latent, time_latent)

        pred, s_pred, s_r_pred, time_pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss, all_loss = self.forward_loss(target, pred, mask)

        # sd_loss = self.forward_sdloss(s_pred, s_r_pred)
        # td_loss = self.forward_tloss(s_pred, time_pred)

        pred = pred.squeeze(2) * mask
        N, L = pred.shape
        n = int(math.sqrt(L))
        pred = pred.reshape(-1, n, n).unsqueeze(1)

        pred = pred + X

        return loss, all_loss, s_loss, t_loss, s_loss, t_loss, pred


def mae_vit_0(patch_size, in_chans, **kwargs):
    model = MaskedAutoencoderViT(
        patch_size=patch_size, embed_dim=16, depth=1, num_heads=2, in_chans=in_chans,
        decoder_embed_dim=16, decoder_depth=1, decoder_num_heads=2,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_1(patch_size, in_chans, **kwargs):
    model = MaskedAutoencoderViT(
        patch_size=patch_size, embed_dim=64, depth=2, num_heads=4, in_chans=in_chans,
        decoder_embed_dim=32, decoder_depth=2, decoder_num_heads=4,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_2(patch_size, in_chans, **kwargs):
    model = MaskedAutoencoderViT(
        patch_size=patch_size, embed_dim=64, depth=6, num_heads=8, in_chans=in_chans,
        decoder_embed_dim=32, decoder_depth=4, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_3(patch_size, in_chans, **kwargs):
    model = MaskedAutoencoderViT(
        patch_size=patch_size, embed_dim=128, depth=4, num_heads=4, in_chans=in_chans,
        decoder_embed_dim=64, decoder_depth=2, decoder_num_heads=4,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_4(patch_size, in_chans, **kwargs):
    model = MaskedAutoencoderViT(
        patch_size=patch_size, embed_dim=256, depth=10, num_heads=8, in_chans=in_chans,
        decoder_embed_dim=128, decoder_depth=5, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_5(patch_size, in_chans, **kwargs):
    model = MaskedAutoencoderViT(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, in_chans=in_chans,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
