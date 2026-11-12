import math
import torch
from torch import nn
from torch.nn import functional as F

def _valid_groups(channels: int, maximum: int=16) -> int:
    groups = min(channels, maximum)
    while channels % groups:
        groups -= 1
    return groups

class _ConvNormAct(nn.Sequential):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int=1, padding: int=0, groups: int=1, dilation: int=1):
        super().__init__(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, groups=groups, dilation=dilation, bias=False), nn.GroupNorm(_valid_groups(out_channels), out_channels), nn.GELU())

def smooth(feature, kernel=5):
    padding = kernel // 2
    return F.avg_pool2d(F.pad(feature, (padding,) * 4, mode='replicate'), kernel, stride=1)

class AdaptiveContext(nn.Module):

    def __init__(self, hidden):
        super().__init__()
        self.branches = nn.ModuleList([_ConvNormAct(hidden, hidden, 3, padding=1, groups=hidden), _ConvNormAct(hidden, hidden, 5, padding=2, groups=hidden), _ConvNormAct(hidden, hidden, 3, padding=3, groups=hidden, dilation=3)])
        self.scale_router = nn.Conv2d(2 * hidden + 1, 3, 1)
        nn.init.zeros_(self.scale_router.weight)
        nn.init.zeros_(self.scale_router.bias)
        self.global_projection = _ConvNormAct(hidden, hidden)
        self.mix = _ConvNormAct(2 * hidden, hidden)

    def forward(self, encoder, decoder, probability):
        joint = 0.5 * (encoder + decoder)
        weights = torch.softmax(self.scale_router(torch.cat((encoder, decoder, probability), dim=1)), dim=1)
        local = sum((weights[:, i:i + 1] * branch(joint) for (i, branch) in enumerate(self.branches)))
        with torch.autocast(device_type=joint.device.type, enabled=False):
            p = probability.float().detach()
            z = joint.float()
            fg = (z * p).sum((-2, -1), keepdim=True) / p.sum((-2, -1), keepdim=True).clamp_min(1e-05)
            bg = (z * (1 - p)).sum((-2, -1), keepdim=True) / (1 - p).sum((-2, -1), keepdim=True).clamp_min(1e-05)
            regional = p * fg + (1 - p) * bg
        regional = self.global_projection(regional.to(joint.dtype))
        return (self.mix(torch.cat((local, regional), dim=1)), weights)

class ADGBCFusion(nn.Module):

    def __init__(self, channels):
        hidden_ratio = 0.25
        correction_scale = 0.1
        prototype_grid = 2
        super().__init__()
        if not 0 < hidden_ratio <= 1 or prototype_grid < 1:
            raise ValueError('invalid hidden ratio or prototype grid')
        self.channels = channels
        self.hidden = min(64, max(16, round(channels * hidden_ratio)))
        self.prototype_grid = prototype_grid
        self.shared_projection = _ConvNormAct(channels, self.hidden)
        self.semantic_head = nn.Conv2d(self.hidden, 1, 1)
        nn.init.normal_(self.semantic_head.weight, std=0.01)
        nn.init.zeros_(self.semantic_head.bias)
        self.parent_gain_logit = nn.Parameter(torch.tensor(-2.0))
        self.prototype_gain_logit = nn.Parameter(torch.tensor(-2.0))
        self.prototype_temperature_raw = nn.Parameter(torch.tensor(1.5))
        self.omni_context = AdaptiveContext(self.hidden)
        cue_channels = 2 * self.hidden + 4
        self.transmission_head = nn.Sequential(_ConvNormAct(cue_channels, self.hidden, 3, padding=1), nn.Conv2d(self.hidden, 2, 1))
        nn.init.normal_(self.transmission_head[-1].weight, std=0.005)
        nn.init.constant_(self.transmission_head[-1].bias, 2.0)
        self.filter_strength_logit = nn.Parameter(torch.tensor(-2.0))
        self.route_head = nn.Sequential(_ConvNormAct(cue_channels, self.hidden, 3, padding=1), nn.Conv2d(self.hidden, 3, 1))
        nn.init.normal_(self.route_head[-1].weight, std=0.005)
        with torch.no_grad():
            self.route_head[-1].bias.copy_(torch.tensor([0.5, 0.0, 0.0]))
        self.context_output = nn.Conv2d(self.hidden, channels, 1, bias=False)
        self.boundary_features = _ConvNormAct(2 * self.hidden + 2, self.hidden, 3, padding=1)
        self.boundary_output = nn.Conv2d(self.hidden, channels, 1, bias=False)
        self.boundary_head = nn.Conv2d(self.hidden, 1, 1)
        for projection in (self.context_output, self.boundary_output):
            nn.init.normal_(projection.weight, std=0.02 / math.sqrt(self.hidden))
        self.context_scale = nn.Parameter(torch.full((1, channels, 1, 1), correction_scale))
        self.boundary_scale = nn.Parameter(torch.full((1, channels, 1, 1), correction_scale))

    def _regional_prior(self, feature, decoder_probability):
        with torch.autocast(device_type=feature.device.type, enabled=False):
            z = F.normalize(feature.float(), dim=1, eps=1e-06)
            probability = decoder_probability.float().detach()
            class_scores = []
            for weight in (probability.pow(4), (1 - probability).pow(4)):
                mass = F.adaptive_avg_pool2d(weight, self.prototype_grid).flatten(2)
                numerator = F.adaptive_avg_pool2d(z * weight, self.prototype_grid).flatten(2)
                prototypes = F.normalize(numerator / mass.clamp_min(1e-06), dim=1, eps=1e-06)
                mixture = (mass + 1e-06) / (mass + 1e-06).sum(-1, keepdim=True)
                scores = torch.bmm(prototypes.transpose(1, 2), z.flatten(2))
                temperature = F.softplus(self.prototype_temperature_raw.float()) + 0.5
                score = torch.logsumexp(temperature * scores + mixture.log().transpose(1, 2), dim=1)
                class_scores.append(score.reshape(z.shape[0], 1, *z.shape[-2:]))
            logit = class_scores[0] - class_scores[1]
            confidence = (2 * probability - 1).abs()
            return (logit.to(feature.dtype), confidence.to(feature.dtype))

    def forward(self, skip, decoder, parent_prior=None):
        if skip.shape[1] != self.channels or decoder.shape[:2] != skip.shape[:2]:
            raise ValueError('incorrect skip/decoder dimensions')
        if decoder.shape[-2:] != skip.shape[-2:]:
            decoder = F.interpolate(decoder, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        encoder_feature = self.shared_projection(skip)
        decoder_feature = self.shared_projection(decoder)
        encoder_prior = self.semantic_head(encoder_feature)
        decoder_prior = self.semantic_head(decoder_feature)
        prior_logit = decoder_prior
        if True and parent_prior is not None:
            prior_logit = prior_logit + torch.sigmoid(self.parent_gain_logit) * F.interpolate(parent_prior, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        probability = torch.sigmoid(prior_logit)
        prototype_logit = torch.zeros_like(prior_logit)
        (prototype_logit, confidence) = self._regional_prior(encoder_feature, probability)
        prior_logit = prior_logit + 0.5 * torch.sigmoid(self.prototype_gain_logit) * confidence * prototype_logit
        semantic_probability = torch.sigmoid(prior_logit)
        with torch.autocast(device_type=skip.device.type, enabled=False):
            p = semantic_probability.float().clamp(1e-06, 1 - 1e-06)
            entropy = -(p * p.log() + (1 - p) * (1 - p).log()) / math.log(2)
        disagreement = (torch.sigmoid(encoder_prior) - torch.sigmoid(decoder_prior)).abs()
        prototype_probability = torch.sigmoid(prototype_logit)
        cues = torch.cat((encoder_feature, decoder_feature, semantic_probability, entropy.to(skip.dtype), disagreement, prototype_probability), dim=1)
        gates = torch.sigmoid(self.transmission_head(cues))
        strength = 0.5 * torch.sigmoid(self.filter_strength_logit)
        low = smooth(skip)
        high = skip - low
        structure_transmission = 1 - strength * (1 - gates[:, 0:1])
        detail_transmission = 1 - strength * (1 - gates[:, 1:2])
        filtered = structure_transmission * low + detail_transmission * high
        route_logits = self.route_head(cues)
        inactive = route_logits.new_tensor((False, not True, not True), dtype=torch.bool)
        route = torch.softmax(route_logits.masked_fill(inactive.view(1, 3, 1, 1), -10000.0), dim=1)
        (context, scale_weights) = self.omni_context(encoder_feature, decoder_feature, probability)
        context_delta = self.context_output(context)
        encoder_high = encoder_feature - smooth(encoder_feature)
        boundary_feature = self.boundary_features(torch.cat((encoder_high, decoder_feature, semantic_probability, disagreement), dim=1))
        boundary_logit = self.boundary_head(boundary_feature)
        boundary_delta = self.boundary_output(boundary_feature)
        evidence = filtered + self.context_scale.tanh() * route[:, 1:2] * context_delta
        evidence = evidence + self.boundary_scale.tanh() * route[:, 2:3] * boundary_delta
        state = {'prior_logit': prior_logit, 'semantic_logits': (encoder_prior, decoder_prior), 'alignment_features': (F.adaptive_avg_pool2d(encoder_feature, 16), F.adaptive_avg_pool2d(decoder_feature, 16)), 'boundary_logit': boundary_logit}
        return (evidence, state)
