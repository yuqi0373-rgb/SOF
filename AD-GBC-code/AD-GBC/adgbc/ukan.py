import torch
from torch import nn
from torch.nn import functional as F
from .backbones.ukan import UKAN
from .fusion import ADGBCFusion

class UKANADGBC(UKAN):

    def __init__(self, num_classes=1, input_channels=3):
        if input_channels != 3:
            raise ValueError('This official U-KAN adapter expects RGB and no deep supervision')
        dims = (128, 160, 256)
        super().__init__(num_classes, input_channels, False, embed_dims=list(dims))
        self.omni3 = ADGBCFusion(dims[0])
        self.omni2 = ADGBCFusion(dims[0] // 4)
        self.omni1 = ADGBCFusion(dims[0] // 8)

    def forward(self, x):
        batch = x.shape[0]
        out = F.relu(F.max_pool2d(self.encoder1(x), 2, 2))
        t1 = out
        out = F.relu(F.max_pool2d(self.encoder2(out), 2, 2))
        t2 = out
        out = F.relu(F.max_pool2d(self.encoder3(out), 2, 2))
        t3 = out
        (out, h, w) = self.patch_embed3(out)
        for block in self.block1:
            out = block(out, h, w)
        out = self.norm3(out).reshape(batch, h, w, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out
        (out, h, w) = self.patch_embed4(out)
        for block in self.block2:
            out = block(out, h, w)
        out = self.norm4(out).reshape(batch, h, w, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.decoder1(out), size=t4.shape[-2:], mode='bilinear', align_corners=False))
        out = out + t4
        (h, w) = out.shape[-2:]
        out = out.flatten(2).transpose(1, 2)
        for block in self.dblock1:
            out = block(out, h, w)
        out = self.dnorm3(out).reshape(batch, h, w, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.decoder2(out), size=t3.shape[-2:], mode='bilinear', align_corners=False))
        (evidence3, state3) = self.omni3(t3, out)
        out = out + evidence3
        (h, w) = out.shape[-2:]
        out = out.flatten(2).transpose(1, 2)
        for block in self.dblock2:
            out = block(out, h, w)
        out = self.dnorm4(out).reshape(batch, h, w, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.decoder3(out), size=t2.shape[-2:], mode='bilinear', align_corners=False))
        (evidence2, state2) = self.omni2(t2, out, parent_prior=state3['prior_logit'])
        out = out + evidence2
        out = F.relu(F.interpolate(self.decoder4(out), size=t1.shape[-2:], mode='bilinear', align_corners=False))
        (evidence1, state1) = self.omni1(t1, out, parent_prior=state2['prior_logit'])
        out = out + evidence1
        out = F.relu(F.interpolate(self.decoder5(out), size=x.shape[-2:], mode='bilinear', align_corners=False))
        states = (state3, state2, state1)
        result = dict(logits=self.final(out), aux_logits=[logit for state in states for logit in state['semantic_logits']], alignment_features=[state['alignment_features'] for state in states], boundary_logits=[state['boundary_logit'] for state in states])
        return result
