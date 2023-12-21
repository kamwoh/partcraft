import argparse
import gc
import os
import re
import shutil

import gradio as gr
import requests
import torch

from dreamcreature.pipeline import create_args, load_pipeline


def download_file(url, local_path):
    if os.path.exists(local_path):
        return

    with requests.get(url, stream=True) as r:
        with open(local_path, 'wb') as f:
            shutil.copyfileobj(r.raw, f)

    # Example usage


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='dreamcreature-sd1.5-cub200')
parser.add_argument('--checkpoint', default='checkpoint-74900')
opt = parser.parse_args()

model_name = opt.model_name
checkpoint_name = opt.checkpoint

repo_url = f"https://huggingface.co/kamwoh/{model_name}/resolve/main"
file_url = repo_url + f"/{checkpoint_name}/pytorch_model.bin"
local_path = f"{model_name}/{checkpoint_name}/pytorch_model.bin"
os.makedirs(f"{model_name}/{checkpoint_name}", exist_ok=True)
download_file(file_url, local_path)

file_url = repo_url + f"/{checkpoint_name}/pytorch_model_1.bin"
local_path = f"{model_name}/{checkpoint_name}/pytorch_model_1.bin"
download_file(file_url, local_path)

OUTPUT_DIR = model_name

args = create_args(OUTPUT_DIR)
if 'dpo' in OUTPUT_DIR:
    args.unet_path = "mhdang/dpo-sd1.5-text2image-v1"

pipe = load_pipeline(args, torch.float16, 'cuda')
pipe = pipe.to(torch.float16)

pipe.verbose = True
pipe.v = 're'
pipe.num_k_per_part = 200

MAPPING = {
    'body': 0,
    'tail': 1,
    'head': 2,
    'wing': 4,
    'leg': 6
}

ID2NAME = open('data/cub200_2011/class_names.txt').readlines()
ID2NAME = [line.strip() for line in ID2NAME]


def process_text(text):
    pattern = r"<([^:>]+):(\d+)>"
    result = text
    offset = 0

    part2id = []

    for match in re.finditer(pattern, text):
        key = match.group(1)
        clsid = int(match.group(2))
        clsid = min(max(clsid, 1), 200)  # must be 1~200

        replacement = f"<{MAPPING[key]}:{clsid - 1}>"
        start, end = match.span()

        # Adjust the start and end positions based on the offset from previous replacements
        start += offset
        end += offset

        # Replace the matched text with the replacement
        result = result[:start] + replacement + result[end:]

        # Update the offset for the next replacement
        offset += len(replacement) - (end - start)

        part2id.append(f'{MAPPING[key]}: {ID2NAME[clsid - 1]}')

    return result, part2id


def generate_images(prompt, negative_prompt, num_inference_steps, guidance_scale, num_images, seed):
    generator = torch.Generator(device='cuda')
    generator = generator.manual_seed(int(seed))

    prompt, part2id = process_text(prompt)
    negative_prompt, _ = process_text(negative_prompt)

    try:
        images = pipe(prompt,
                      negative_prompt=negative_prompt, generator=generator,
                      num_inference_steps=int(num_inference_steps), guidance_scale=guidance_scale,
                      num_images_per_prompt=num_images).images
    except Exception as e:
        raise gr.Error(f"Probably due to the prompt have invalid input, please follow the instruction. "
                       f"The error message: {e}")
    finally:
        gc.collect()
        torch.cuda.empty_cache()

    return images, '; '.join(part2id)


with gr.Blocks(title="DreamCreature") as demo:
    with gr.Row():
        gr.Markdown(
            """
            # DreamCreature (CUB-200-2011)
            To create your own creature, you can type:

            `"a photo of a <head:id> <wing:id> bird"` where `id` ranges from 1~200 (200 classes corresponding to CUB-200-2011)

            For instance `"a photo of a <head:17> <wing:18> bird"` using head of `cardinal (17)` and wing of `spotted catbird (18)`

            Please see `id` in https://github.com/kamwoh/dreamcreature/blob/master/src/data/cub200_2011/class_names.txt

            You can also try any prompt you like such as:

            Sub-concept transfer: `"a photo of a <wing:17> cat"`

            Inspiring design: `"a photo of a <head:101> <wing:191> teddy bear"`

            (Experimental) You can also use two parts together such as:

            `"a photo of a <head:17> <head:18> bird"` mixing head of `cardinal (17)` and `spotted catbird (18)`

            The current available parts are: `head`, `body`, `wing`, `tail`, and `leg`

            """)
    with gr.Column():
        with gr.Row():
            with gr.Group():
                prompt = gr.Textbox(label="Prompt", value="a photo of a <head:16> <wing:17> bird")
                negative_prompt = gr.Textbox(label="Negative Prompt",
                                             value="blurry, ugly, duplicate, poorly drawn, deformed, mosaic")
                num_inference_steps = gr.Slider(minimum=10, maximum=100, step=1, value=30, label="Num Inference Steps")
                guidance_scale = gr.Slider(minimum=2, maximum=20, step=0.1, value=7.5, label="Guidance Scale")
                num_images = gr.Slider(minimum=1, maximum=4, step=1, value=4, label="Number of Images")
                seed = gr.Number(label="Seed", value=42)
                button = gr.Button()

            with gr.Column():
                output_images = gr.Gallery(columns=4, label='Output')
                markdown_labels = gr.Markdown("")

    button.click(fn=generate_images,
                 inputs=[prompt, negative_prompt, num_inference_steps, guidance_scale, num_images,
                         seed], outputs=[output_images, markdown_labels], show_progress=True)

demo.queue().launch(inline=False, share=False, debug=True, server_name='0.0.0.0')
