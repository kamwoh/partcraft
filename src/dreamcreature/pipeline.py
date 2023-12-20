import os

from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import Attention
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import *
from omegaconf import OmegaConf

from dreamcreature.attn_processor import LoRAAttnProcessorCustom
from dreamcreature.mapper import TokenMapper
from dreamcreature.text_encoder import CustomCLIPTextModel
from dreamcreature.tokenizer import MultiTokenCLIPTokenizer
from utils import add_tokens, get_attn_processors


def setup_attn_processor(unet, **kwargs):
    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        lora_attn_procs[name] = LoRAAttnProcessorCustom(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=kwargs['rank'],
        )

    unet.set_attn_processor(lora_attn_procs)


def load_attn_processor(unet, filename):
    lora_layers = AttnProcsLayers(get_attn_processors(unet))
    lora_layers.load_state_dict(torch.load(filename))


def convert_prompt(prompt: str, replace_token: bool = False, v1=True):
    r"""
    Parameters:
        prompt (`str`):
            The prompt to guide the image generation.

    Returns:
        `str`: The converted prompt
    """
    if ':' not in prompt:
        return prompt, None, None

    splits = prompt.replace('.', '').strip().split(' ')
    # v1: a photo of a 0:1 1:24 ...
    # v2: a photo of a <0:1> <1:24> ...
    ints = []
    noncode_start = ''
    noncode_end = ''
    parts = ''
    parts_i = []
    split_tokens = []
    for b in splits:
        if ':' not in b:
            split_tokens.append(b)
            continue

        if v1:
            i, b = b.strip().split(':')
            has_comma = ',' in b
            if has_comma:
                b = b[:-1]
            intb = int(b)
            parts += f'<part>_{i} '
            split_tokens.append(f'<part>_{i}')
            if has_comma:
                split_tokens.append(',')
        else:
            if b[0] == '<':
                if '>' not in b:  # no closing >, ignore
                    split_tokens.append(b)
                    continue

                i, b = b[1:].strip().split(':')
                token_to_add = ''
                if b[-1] in [',', '.']:
                    token_to_add = b[-1]
                    b = b[:-1]

                if b[-1] == '>':
                    b = b[:-1]
                else:  # not >, just search for the first >
                    for ci, char in enumerate(b):
                        if char == '>':
                            token_to_add = b[ci + 1:] + token_to_add
                            b = b[:ci]  # skip >
                            break
            else:  # has : but not start with <
                split_tokens.append(b)
                continue

            intb = abs(int(b))  # just force negative one to positive

            parts += f'<part>_{i} '
            split_tokens.append(f'<part>_{i}')
            if len(token_to_add) != 0:
                split_tokens.append(token_to_add)

        try:
            int(i)
        except:
            raise ValueError(f'cannot cast `part` properly, please make sure input is correct')

        parts_i.append(int(i))
        ints.append(intb)

    ints = torch.tensor(ints)  # (nparts,)

    if replace_token:
        new_caption = f'{noncode_start.strip()} <part> {noncode_end.strip()}'
    else:
        new_caption = ' '.join(split_tokens)

    new_caption = new_caption.strip()

    return new_caption, ints, parts_i


class DreamCreatureSDPipeline(StableDiffusionPipeline):
    def _maybe_convert_prompt(self, prompt: str, tokenizer: MultiTokenCLIPTokenizer):
        r"""
        Maybe convert a prompt into a "multi vector"-compatible prompt. If the prompt includes a token that corresponds
        to a multi-vector textual inversion embedding, this function will process the prompt so that the special token
        is replaced with multiple special tokens each corresponding to one of the vectors. If the prompt has no textual
        inversion token or a textual inversion token that is a single vector, the input prompt is simply returned.

        Parameters:
            prompt (`str`):
                The prompt to guide the image generation.
            tokenizer (`PreTrainedTokenizer`):
                The tokenizer responsible for encoding the prompt into input tokens.

        Returns:
            `str`: The converted prompt
        """
        if hasattr(self, 'replace_token'):
            replace_token = self.replace_token
        else:
            replace_token = True

        if hasattr(self, 'v1'):
            v1 = self.v1
        else:
            v1 = True

        new_caption, code, parts_i = convert_prompt(prompt, replace_token, v1)
        if hasattr(self, 'num_k_per_part'):
            if code is not None and any(code >= self.num_k_per_part):
                raise ValueError(f'`id` cannot more than {self.num_k_per_part}')

        if hasattr(self, 'verbose') and self.verbose:
            print(new_caption)

        return new_caption, code, parts_i

    def compute_prompt_embeddings(self, prompts, device):
        # textual inversion: procecss multi-vector tokens if necessary
        if not isinstance(prompts, List):
            prompts = [prompts]

        prompt_embeds_concat = []
        for prompt in prompts:
            prompt, code, parts_i = self.maybe_convert_prompt(prompt, self.tokenizer)

            if hasattr(self, 'replace_token'):
                replace_token = self.replace_token
            else:
                replace_token = True

            text_inputs = self.tokenizer(
                prompt,
                replace_token=replace_token,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            if hasattr(self, 'verbose') and self.verbose:
                print(text_input_ids)

            untruncated_ids = self.tokenizer(prompt,
                                             replace_token=replace_token,
                                             padding="longest",
                                             return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                    text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1: -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            if code is None:
                modified_hs = None
            else:
                placeholder_token_ids = self.placeholder_token_ids
                placeholder_token_ids = [placeholder_token_ids[i] for i in parts_i]  # follow the order of prompt's i
                mapper_outputs = self.simple_mapper(code.unsqueeze(0).to(device), torch.tensor(parts_i).to(device))
                modified_hs = self.text_encoder.text_model.forward_embeddings_with_mapper(text_input_ids.to(device),
                                                                                          None,
                                                                                          mapper_outputs,
                                                                                          placeholder_token_ids)

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
                hidden_states=modified_hs
            )
            prompt_embeds = prompt_embeds[0]
            prompt_embeds_concat.append(prompt_embeds)

        if len(prompt_embeds_concat) == 1:
            return prompt_embeds_concat[0]
        else:
            return torch.cat(prompt_embeds_concat, dim=0)

    def encode_prompt(
            self,
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            lora_scale: Optional[float] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self.compute_prompt_embeddings(prompt, device)

        # if self.text_encoder is not None:
        #     prompt_embeds_dtype = self.text_encoder.dtype
        # elif self.unet is not None:
        #     prompt_embeds_dtype = self.unet.dtype
        # else:
        #     prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds_dtype = self.unet.dtype  # should be unet only because this is unet's condition input
        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt] * batch_size
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            negative_prompt_embeds = []
            for u_tokens in uncond_tokens:
                negative_prompt_embeds.append(self.compute_prompt_embeddings(u_tokens, device))
            negative_prompt_embeds = torch.cat(negative_prompt_embeds, dim=0)

            # if isinstance(self, TextualInversionLoaderMixin):
            #     uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            # max_length = prompt_embeds.shape[1]
            # uncond_input = self.tokenizer(
            #     uncond_tokens,
            #     padding="max_length",
            #     max_length=max_length,
            #     truncation=True,
            #     return_tensors="pt",
            # )
            #
            # if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            #     attention_mask = uncond_input.attention_mask.to(device)
            # else:
            #     attention_mask = None
            #
            # negative_prompt_embeds = self.text_encoder(
            #     uncond_input.input_ids.to(device),
            #     attention_mask=attention_mask,
            # )
            # negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds, negative_prompt_embeds

    @torch.no_grad()
    def obtain_attention_map(self, image, prompt, timesteps):
        prompt, codes, index = self.maybe_convert_prompt(prompt, self.tokenizer)

        if hasattr(self, 'replace_token'):
            replace_token = self.replace_token
        else:
            replace_token = True

        text_inputs = self.tokenizer(
            prompt,
            replace_token=replace_token,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids

        placeholder_token_ids = self.placeholder_token_ids
        placeholder_token_ids = [placeholder_token_ids[i] for i in index]

        # forward an image, denoise it and obtain the attention map
        device = self._execution_device

        latents = self.vae.encode(image.to(device, dtype=self.weight_dtype)).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        # bsz = latents.shape[0]
        # Sample a random timestep for each image
        # timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long().to(latents.device)

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noise = torch.randn_like(latents)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        mapper_outputs = self.simple_mapper(codes.unsqueeze(0).to(device), torch.tensor(index).to(device))
        # print(mapper_outputs.size(), batch["input_ids"].size())
        modified_hs = self.text_encoder.text_model.forward_embeddings_with_mapper(input_ids.to(device),
                                                                                  None,
                                                                                  mapper_outputs,
                                                                                  placeholder_token_ids)
        encoder_hidden_states = self.text_encoder(input_ids, hidden_states=modified_hs)[0]
        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states.to(dtype=self.weight_dtype)).sample

        attn_probs = {}

        for name, module in self.unet.named_modules():
            if isinstance(module, Attention) and module.attn_probs is not None:
                a = module.attn_probs[0].mean(dim=0)  # (2,Head,H,W,77)->(H,W,77)
                attn_probs[name] = a

        avg_attn_map = []
        for name in attn_probs:
            avg_attn_map.append(attn_probs[name])
        avg_attn_map = torch.stack(avg_attn_map, dim=0).mean(dim=0)  # (5,B,H,W,77) -> (B,H,W,77)

        return attn_probs, avg_attn_map, input_ids

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            guidance_rescale: float = 0.0,
            get_attention_map: bool = False
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.7):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            if get_attention_map:
                attn_maps = {}  # each t one attn map

            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]
                if get_attention_map:
                    attn_probs = {}

                    for name, module in self.unet.named_modules():
                        if isinstance(module, Attention) and module.attn_probs is not None:
                            a = module.attn_probs[1].mean(dim=0)  # (2,Head,H,W,77)->(H,W,77)
                            attn_probs[name] = a

                    attn_maps[i] = attn_probs

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if get_attention_map:
            output_maps = {}
            for name in attn_probs.keys():
                timeavg_maps = []
                for i in attn_maps.keys():
                    timeavg_maps.append(attn_maps[i][name])
                timeavg_maps = torch.stack(timeavg_maps, dim=0).mean(dim=0)
                output_maps[name] = timeavg_maps

            avg_attn_map = []
            for name in attn_probs:
                avg_attn_map.append(attn_probs[name])
            avg_attn_map = torch.stack(avg_attn_map, dim=0).mean(dim=0)  # (5,B,H,W,77) -> (B,H,W,77)
            output_maps['avg'] = avg_attn_map

            del attn_maps
            del attn_probs
            self.attn_maps = output_maps

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)


def create_args(output_dir, num_parts=8, num_k_per_part=256):
    args = OmegaConf.create({
        'pretrained_model_name_or_path': 'runwayml/stable-diffusion-v1-5',
        'num_parts': num_parts,
        'num_k_per_part': num_k_per_part,
        'revision': None,
        'variant': None,
        'rank': 4,
        'projection_nlayers': 1,
        'output_dir': output_dir
    })
    folders = sorted(os.listdir(args.output_dir))
    cps = [int(f.split('-')[1]) for f in folders if 'checkpoint' in f and '.ipynb' not in f]
    maxcp = max(cps)

    args.maxcp = maxcp
    args.unet_path = None
    return args


def load_pipeline(args, weight_dtype=torch.float16, device=torch.device('cuda')):
    tokenizer = MultiTokenCLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CustomCLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    unet_path = args.unet_path if args.unet_path is not None else args.pretrained_model_name_or_path
    unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
        unet_path, subfolder="unet", revision=args.revision
    )
    pipeline = DreamCreatureSDPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=unet,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    pipeline.num_k_per_part = args.num_k_per_part
    pipeline.num_parts = args.num_parts
    placeholder_token = "<part>"
    initializer_token = None
    placeholder_token_ids = add_tokens(tokenizer,
                                       text_encoder,
                                       placeholder_token,
                                       args.num_parts,
                                       initializer_token)
    pipeline.placeholder_token_ids = placeholder_token_ids
    pipeline.simple_mapper = TokenMapper(args.num_parts,
                                         args.num_k_per_part,
                                         768,
                                         args.projection_nlayers)
    pipeline.simple_mapper.load_state_dict(torch.load(args.output_dir + f'/checkpoint-{args.maxcp}/pytorch_model_1.bin',
                                                      map_location='cpu'))
    pipeline.simple_mapper.to(device)
    pipeline.replace_token = False
    pipeline = pipeline.to(device)

    # load attention processors
    # pipeline.unet.load_attn_procs(args.output_dir, use_safetensors=not args.custom_diffusion)
    setup_attn_processor(pipeline.unet, rank=args.rank)
    load_attn_processor(pipeline.unet, args.output_dir + f'/checkpoint-{args.maxcp}/pytorch_model.bin')
    return pipeline
