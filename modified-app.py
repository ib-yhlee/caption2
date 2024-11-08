import spaces
import gradio as gr
from huggingface_hub import InferenceClient
from torch import nn
from transformers import AutoModel, AutoProcessor, AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast, AutoModelForCausalLM
from pathlib import Path
import torch
import torch.amp.autocast_mode
from PIL import Image
import os
import torchvision.transforms.functional as TVF

# ... (기존의 상수, 클래스 정의 등은 동일하게 유지)

@spaces.GPU()
@torch.no_grad()
def stream_chat(input_images: list[Image.Image], caption_type: str, caption_length: str | int, extra_options: list[str], name_input: str, custom_prompt: str) -> tuple[str, dict]:
    torch.cuda.empty_cache()
    
    # Dictionary to store results for each image
    results = {}
    
    for img in input_images:
        # Get the original filename without extension
        img_filename = getattr(img, 'orig_name', 'untitled').rsplit('.', 1)[0]
        
        # 'any' means no length specified
        length = None if caption_length == "any" else caption_length

        if isinstance(length, str):
            try:
                length = int(length)
            except ValueError:
                pass
        
        # Build prompt
        if length is None:
            map_idx = 0
        elif isinstance(length, int):
            map_idx = 1
        elif isinstance(length, str):
            map_idx = 2
        else:
            raise ValueError(f"Invalid caption length: {length}")
        
        prompt_str = CAPTION_TYPE_MAP[caption_type][map_idx]

        # Add extra options
        if len(extra_options) > 0:
            prompt_str += " " + " ".join(extra_options)
        
        # Add name, length, word_count
        prompt_str = prompt_str.format(name=name_input, length=caption_length, word_count=caption_length)

        if custom_prompt.strip() != "":
            prompt_str = custom_prompt.strip()
        
        # For debugging
        print(f"Processing image: {img_filename}")
        print(f"Prompt: {prompt_str}")

        # Preprocess image
        image = img.resize((384, 384), Image.LANCZOS)
        pixel_values = TVF.pil_to_tensor(image).unsqueeze(0) / 255.0
        pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])
        pixel_values = pixel_values.to('cuda')

        # Embed image
        with torch.amp.autocast_mode.autocast('cuda', enabled=True):
            vision_outputs = clip_model(pixel_values=pixel_values, output_hidden_states=True)
            embedded_images = image_adapter(vision_outputs.hidden_states)
            embedded_images = embedded_images.to('cuda')
        
        # Build the conversation
        convo = [
            {
                "role": "system",
                "content": "You are a helpful image captioner.",
            },
            {
                "role": "user",
                "content": prompt_str,
            },
        ]

        # Format the conversation
        convo_string = tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
        assert isinstance(convo_string, str)

        # Tokenize the conversation
        convo_tokens = tokenizer.encode(convo_string, return_tensors="pt", add_special_tokens=False, truncation=False)
        prompt_tokens = tokenizer.encode(prompt_str, return_tensors="pt", add_special_tokens=False, truncation=False)
        assert isinstance(convo_tokens, torch.Tensor) and isinstance(prompt_tokens, torch.Tensor)
        convo_tokens = convo_tokens.squeeze(0)
        prompt_tokens = prompt_tokens.squeeze(0)

        # Calculate where to inject the image
        eot_id_indices = (convo_tokens == tokenizer.convert_tokens_to_ids("<|eot_id|>")).nonzero(as_tuple=True)[0].tolist()
        assert len(eot_id_indices) == 2, f"Expected 2 <|eot_id|> tokens, got {len(eot_id_indices)}"

        preamble_len = eot_id_indices[1] - prompt_tokens.shape[0]

        # Embed the tokens
        convo_embeds = text_model.model.embed_tokens(convo_tokens.unsqueeze(0).to('cuda'))

        # Construct the input
        input_embeds = torch.cat([
            convo_embeds[:, :preamble_len],
            embedded_images.to(dtype=convo_embeds.dtype),
            convo_embeds[:, preamble_len:],
        ], dim=1).to('cuda')

        input_ids = torch.cat([
            convo_tokens[:preamble_len].unsqueeze(0),
            torch.zeros((1, embedded_images.shape[1]), dtype=torch.long),
            convo_tokens[preamble_len:].unsqueeze(0),
        ], dim=1).to('cuda')
        attention_mask = torch.ones_like(input_ids)

        # Generate caption
        generate_ids = text_model.generate(
            input_ids, 
            inputs_embeds=input_embeds, 
            attention_mask=attention_mask, 
            max_new_tokens=300, 
            do_sample=True, 
            suppress_tokens=None
        )

        # Trim off the prompt
        generate_ids = generate_ids[:, input_ids.shape[1]:]
        if generate_ids[0][-1] == tokenizer.eos_token_id or generate_ids[0][-1] == tokenizer.convert_tokens_to_ids("<|eot_id|>"):
            generate_ids = generate_ids[:, :-1]

        caption = tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
        
        # Save caption to file
        output_path = f"{img_filename}.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(caption.strip())
        
        # Store results
        results[img_filename] = caption.strip()

    return prompt_str, results

with gr.Blocks() as demo:
    gr.HTML(TITLE)

    with gr.Row():
        with gr.Column():
            # Changed to accept multiple images
            input_images = gr.File(
                file_count="multiple",
                file_types=["image"],
                type="file",
                label="Input Images"
            )

            caption_type = gr.Dropdown(
                choices=["Descriptive", "Descriptive (Informal)", "Training Prompt", "MidJourney", "Booru tag list", "Booru-like tag list", "Art Critic", "Product Listing", "Social Media Post"],
                label="Caption Type",
                value="Descriptive",
            )

            caption_length = gr.Dropdown(
                choices=["any", "very short", "short", "medium-length", "long", "very long"] +
                        [str(i) for i in range(20, 261, 10)],
                label="Caption Length",
                value="long",
            )

            extra_options = gr.CheckboxGroup(
                choices=[
                    "If there is a person/character in the image you must refer to them as {name}.",
                    "Do NOT include information about people/characters that cannot be changed (like ethnicity, gender, etc), but do still include changeable attributes (like hair style).",
                    # ... (나머지 옵션들은 동일하게 유지)
                ],
                label="Extra Options"
            )

            name_input = gr.Textbox(label="Person/Character Name (if applicable)")
            gr.Markdown("**Note:** Name input is only used if an Extra Option is selected that requires it.")

            custom_prompt = gr.Textbox(label="Custom Prompt (optional, will override all other settings)")
            gr.Markdown("**Note:** Alpha Two is not a general instruction follower and will not follow prompts outside its training data well. Use this feature with caution.")

            run_button = gr.Button("Caption")
        
        with gr.Column():
            output_prompt = gr.Textbox(label="Prompt that was used")
            output_captions = gr.JSON(label="Generated Captions")
            gr.Markdown("**Note:** Captions are also saved as individual text files with the same name as the input images.")
    
    run_button.click(
        fn=stream_chat,
        inputs=[input_images, caption_type, caption_length, extra_options, name_input, custom_prompt],
        outputs=[output_prompt, output_captions]
    )

if __name__ == "__main__":
    demo.launch()
