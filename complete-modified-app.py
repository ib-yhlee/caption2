with gr.Blocks() as demo:
    gr.HTML(TITLE)

    with gr.Row():
        with gr.Column():
            # Changed to accept multiple files
            input_files = gr.File(
                file_count="multiple",
                file_types=["image"],
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
                    "Include information about lighting.",
                    "Include information about camera angle.",
                    "Include information about whether there is a watermark or not.",
                    "Include information about whether there are JPEG artifacts or not.",
                    "If it is a photo you MUST include information about what camera was likely used and details such as aperture, shutter speed, ISO, etc.",
                    "Do NOT include anything sexual; keep it PG.",
                    "Do NOT mention the image's resolution.",
                    "You MUST include information about the subjective aesthetic quality of the image from low to very high.",
                    "Include information on the image's composition style, such as leading lines, rule of thirds, or symmetry.",
                    "Do NOT mention any text that is in the image.",
                    "Specify the depth of field and whether the background is in focus or blurred.",
                    "If applicable, mention the likely use of artificial or natural lighting sources.",
                    "Do NOT use any ambiguous language.",
                    "Include whether the image is sfw, suggestive, or nsfw.",
                    "ONLY describe the most important elements of the image."
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
        inputs=[input_files, caption_type, caption_length, extra_options, name_input, custom_prompt],
        outputs=[output_prompt, output_captions]
    )

if __name__ == "__main__":
    demo.launch()
