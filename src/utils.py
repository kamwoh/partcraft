import torch


def add_tokens(tokenizer, text_encoder, placeholder_token, num_vec_per_token=1, initializer_token=None):
    """
    Add tokens to the tokenizer and set the initial value of token embeddings
    """
    tokenizer.add_placeholder_tokens(placeholder_token, num_vec_per_token=num_vec_per_token)
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeds = text_encoder.get_input_embeddings().weight.data
    placeholder_token_ids = tokenizer.encode(placeholder_token, add_special_tokens=False)
    if initializer_token:
        token_ids = tokenizer.encode(initializer_token, add_special_tokens=False)
        for i, placeholder_token_id in enumerate(placeholder_token_ids):
            token_embeds[placeholder_token_id] = token_embeds[token_ids[i * len(token_ids) // num_vec_per_token]]
    else:
        for i, placeholder_token_id in enumerate(placeholder_token_ids):
            token_embeds[placeholder_token_id] = torch.randn_like(token_embeds[placeholder_token_id])
    return placeholder_token_ids


def tokenize_prompt(tokenizer, prompt, replace_token=False):
    text_inputs = tokenizer(
        prompt,
        replace_token=replace_token,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids
