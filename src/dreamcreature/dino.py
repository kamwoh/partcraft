import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor


class DINO(nn.Module):
    NAMES = {
        'dino': 'facebook/dino-vits16',
        'dinov2': 'facebook/dinov2-base'
    }

    def __init__(self, name='facebook/dinov2-base', **kwargs):
        super().__init__()

        self.model = AutoModel.from_pretrained(name)
        self.processor = AutoImageProcessor.from_pretrained(name)

    def forward(self, image):
        vit_output = self.model(image,
                                output_hidden_states=True,
                                return_dict=True)

        outputs = {}
        for i in range(1, len(vit_output.hidden_states)):
            outputs[f'block{i}'] = vit_output.hidden_states[i][:, 0]  # get cls only
        outputs['feats'] = outputs[f'block{i}']
        return outputs

    def preprocess(self, image, size=None):
        inputs = self.processor(images=image, return_tensors="pt", size=size)
        return inputs['pixel_values']

    def get_feat_maps(self, image, index=-1):
        vit_output = self.model(image,
                                output_hidden_states=True,
                                return_dict=True)

        last_hidden_states = vit_output.hidden_states[index]

        B, T, C = last_hidden_states.size()
        HW = int((T - 1) ** 0.5)

        return last_hidden_states[:, 1:, :].reshape(B, HW, HW, C).permute(0, 3, 1, 2)  # (B, C, H, W)
