import os
from typing import List

import numpy as np
import torch
import torchpq


class KMeansSegmentation:
    FOREGROUND = 'foreground_background'
    COARSE = 'coarse_kmeans'
    FINE = 'fine_kmeans'

    def __init__(self, path, foreground_idx=0, background_code=7, M=8, K=256):
        if not os.path.exists(path):
            raise FileNotFoundError(f'please train {path}')
        kmeans = torch.load(path)

        self.foreground_idx = foreground_idx
        self.kmeans = kmeans
        self.background_code = background_code
        self.M = M
        self.K = K

        self.fg: torchpq.clustering.KMeans = kmeans[KMeansSegmentation.FOREGROUND]
        self.coarse: torchpq.clustering.KMeans = kmeans[KMeansSegmentation.COARSE]
        self.fine: List[torchpq.clustering.KMeans] = kmeans[KMeansSegmentation.FINE]

    def obtain_fine_feats(self, prompt, filter_idxs=[]):
        if isinstance(prompt, str):
            code = np.zeros((self.M,), dtype=int)
            splits = prompt.strip().split(' ')
            for s in splits:
                m, k = s.split(':')
                code[int(m)] = int(k)
        else:
            code = prompt

        fine_feats = []
        for m in range(self.M):
            fine_feats.append(self.fine[m].centroids.cpu().t()[code[m]])
        fine_feats = torch.stack(fine_feats, dim=0)

        if len(filter_idxs) != 0:
            new_fine_feats = []

            for m in range(self.M):
                if m not in filter_idxs:
                    new_fine_feats.append(fine_feats[m])

            fine_feats = torch.stack(new_fine_feats, dim=0)

        return fine_feats

    def get_segmask(self, feat_map, with_appeared_tokens=False):
        N, C, H, W = feat_map.size()
        query = feat_map.cuda().reshape(N, C, H * W).permute(0, 2, 1)  # (N, H*W, C)

        fg_labels = self.fg.predict(query.reshape(N * H * W, C).t().contiguous())  # (N*H*W)
        fg_labels = fg_labels.reshape(N, H * W)

        fg_idx = self.foreground_idx
        bg_idx = 1 - self.foreground_idx

        nobg = []
        bgmean = []

        for i in range(N):
            bgnorm_mean = query[i][fg_labels[i] == bg_idx].mean(dim=0, keepdim=True)

            if fg_idx == 0:
                bg_mask = fg_labels[i]
            else:
                bg_mask = 1 - fg_labels[i]

            bg_mask = bg_mask.unsqueeze(1)
            nobg.append(query[i] * (1 - bg_mask) + (-1 * bg_mask))
            bgmean.append(bgnorm_mean)

        nobg = torch.stack(nobg, dim=0)  # (B, H*W, C)
        coarse_labels = self.coarse.predict(nobg.reshape(N * H * W, 768).t().contiguous())
        coarse_labels = coarse_labels.reshape(N, H, W)

        segmasks = []
        for m in range(self.M):
            mask = (coarse_labels == m).float()  # (N, H, W)
            segmasks.append(mask)
        segmasks = torch.stack(segmasks, dim=1)  # (N, M, H, W)

        if with_appeared_tokens:
            appeared_tokens = []
            for i in range(N):
                appeared_tokens.append(torch.unique(coarse_labels[i].reshape(-1)))
            return appeared_tokens

        return segmasks

    def predict(self, feat_map, disable=True, filter_idxs=[]):
        # feat_map: (B, C, H, W)

        N, C, H, W = feat_map.size()
        query = feat_map.reshape(N, C, H * W).permute(0, 2, 1)  # (N, H*W, C)

        fg_labels = self.fg.predict(query.reshape(N * H * W, C).t().contiguous().cuda()).cpu()  # (N*H*W)
        fg_labels = fg_labels.reshape(N, H * W)

        fg_idx = self.foreground_idx
        bg_idx = 1 - self.foreground_idx

        nobg = []
        bgmean = []

        for i in range(N):
            bgnorm_mean = query[i][fg_labels[i] == bg_idx].mean(dim=0, keepdim=True)

            if fg_idx == 0:
                bg_mask = fg_labels[i]
            else:
                bg_mask = 1 - fg_labels[i]

            bg_mask = bg_mask.unsqueeze(1)
            nobg.append(query[i] * (1 - bg_mask) + (-1 * bg_mask))
            bgmean.append(bgnorm_mean)

        nobg = torch.stack(nobg, dim=0)  # (B, H*W, C)
        bgmean = torch.cat(bgmean, dim=0)

        coarse_labels = self.coarse.predict(nobg.reshape(N * H * W, 768).t().contiguous().cuda()).cpu()
        coarse_labels = coarse_labels.reshape(N, H * W)

        from tqdm.auto import tqdm

        fgmean = []
        M = self.M

        locs = np.zeros((N, M, 2))

        for i in tqdm(range(N), disable=disable):
            mean_feats = []
            for m in range(M):
                coarse_mask = coarse_labels[i] == m
                if coarse_mask.sum().item() == 0:
                    m_mean_feats = torch.zeros(1, C)
                else:
                    locs[i, m] = (coarse_mask.reshape(H, W).nonzero().float().add(0.5).mean(dim=0) / H).cpu().numpy()
                    m_mean_feats = query[i][coarse_mask].mean(dim=0, keepdim=True)  # (H*W,C) -> (1,C)

                mean_feats.append(m_mean_feats)

            mean_feats = torch.cat(mean_feats, dim=0)
            fgmean.append(mean_feats)

        fgmean = torch.stack(fgmean, dim=0)  # (N, M, C)
        final_labels = torch.ones(N, M) * self.K

        for m in range(M):
            fine_kmeans = self.fine[m]

            if m == self.background_code:
                fine_labels = fine_kmeans.predict(bgmean.t().contiguous().cuda()).cpu()
                final_labels[:, m] = fine_labels
            else:
                fine_inp = fgmean[:, m].reshape(N, C)
                is_zero = fine_inp.sum(dim=1) == 0
                fine_labels = fine_kmeans.predict(fine_inp.t().contiguous().cuda()).cpu()
                fine_labels[is_zero] = self.K

                final_labels[:, m] = fine_labels

        fgmean[:, self.background_code] = bgmean
        fine_prompts = []

        for fine_label in final_labels:
            prompt_dict = {k: int(v) for k, v in enumerate(list(fine_label))}
            if len(filter_idxs) != 0:
                for i in filter_idxs:
                    del prompt_dict[i]
            prompt = ' '.join([f'{k}:{v}' for k, v in prompt_dict.items() if v != self.K])
            fine_prompts.append(prompt)

        return {
            'features': fgmean,
            'fg_labels': fg_labels,
            'coarse_labels': coarse_labels,
            'fine_labels': final_labels,
            'fine_prompts': fine_prompts,
            'location': locs
        }
