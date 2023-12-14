import torch
import torch.nn.functional as F
from diffusers import UNet2DConditionModel
from diffusers.models.attention_processor import Attention

from dreamcreature.dino import DINO
from dreamcreature.kmeans_segmentation import KMeansSegmentation


def dreamcreature_loss(batch,
                       unet: UNet2DConditionModel,
                       dino: DINO,
                       seg: KMeansSegmentation,
                       placeholder_token_ids,
                       accelerator):
    attn_probs = {}

    for name, module in unet.named_modules():
        if isinstance(module, Attention) and module.attn_probs is not None:
            a = module.attn_probs.mean(dim=1)  # (B,Head,H,W,77) -> (B,H,W,77)
            attn_probs[name] = a

    avg_attn_map = []
    for name in attn_probs:
        avg_attn_map.append(attn_probs[name])

    avg_attn_map = torch.stack(avg_attn_map, dim=0).mean(dim=0)  # (L,B,H,W,77) -> (B,H,W,77)
    B, H, W, seq_length = avg_attn_map.size()
    located_attn_map = []

    # locate the attn map
    for i, placeholder_token_id in enumerate(placeholder_token_ids):
        for bi in range(B):
            if "input_ids" in batch:
                learnable_idx = (batch["input_ids"][bi] == placeholder_token_id).nonzero(as_tuple=True)[0]
            else:
                learnable_idx = (batch["input_ids_one"][bi] == placeholder_token_id).nonzero(as_tuple=True)[0]

            if len(learnable_idx) != 0:  # only assign if found
                if len(learnable_idx) == 1:
                    offset_learnable_idx = learnable_idx
                else:  # if there is two and above.
                    raise NotImplementedError

                located_map = avg_attn_map[bi, :, :, offset_learnable_idx]
                located_attn_map.append(located_map)
            else:
                located_attn_map.append(torch.zeros(H, W, 1).to(accelerator.device))

    M = len(placeholder_token_ids)
    located_attn_map = torch.stack(located_attn_map, dim=0).reshape(M, B, H, W).transpose(0, 1)  # (B, M, 16, 16)

    raw_images = batch['raw_images']
    dino_input = dino.preprocess(raw_images, size=224)
    with torch.no_grad():
        dino_ft = dino.get_feat_maps(dino_input)
        segmasks, appeared_tokens = seg.get_segmask(dino_ft, True)  # (B, M, H, W)
        segmasks = segmasks.to(located_attn_map.dtype)
        if H != 16:  # for res 1024
            segmasks = F.interpolate(segmasks, (H, W), mode='nearest')

        masks = []
        for i, appeared in enumerate(appeared_tokens):
            mask = (segmasks[i, appeared].sum(dim=0) > 0).float()  # (A, H, W) -> (H, W)
            masks.append(mask)
        masks = torch.stack(masks, dim=0)  # (B, H, W)
        batch['masks'] = masks

    norm_map = located_attn_map / located_attn_map.sum(dim=1, keepdim=True).clamp(min=1e-6)
    # if norm_map is assigned manually, means the sub-concept token is not found, hence no gradient will be backprop
    attn_loss = F.binary_cross_entropy(norm_map.clamp(min=0, max=1),
                                       segmasks.clamp(min=0, max=1))
    return attn_loss, located_attn_map.detach().max()
