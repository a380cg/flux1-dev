import streamlit as st
import torch
from diffusers import Flux1InpaintPipeline, ControlNetModel  # Adjusted class names
from diffusers.utils import load_image
from PIL import Image
import numpy as np
import io

# Set device to CPU explicitly
device = "cpu"

# Load models (use float32 for CPU compatibility)
@st.cache_resource
def load_pipeline():
    controlnet = ControlNetModel.from_pretrained(
        "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta",
        torch_dtype=torch.float32
    ).to(device)
    # Use a generic inpainting pipeline if FluxControlNetInpaintingPipeline isn’t available
    pipe = Flux1InpaintPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        controlnet=controlnet,
        torch_dtype=torch.float32
    ).to(device)
    return pipe

# Load the pipeline once and cache it
pipe = load_pipeline()

# Streamlit UI
st.title("FLUX.1 Inpainting Demo")
st.write("Upload an image and a mask, then enter a prompt to inpaint the masked area.")

# File uploaders
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
uploaded_mask = st.file_uploader("Upload a mask (white = inpaint, black = preserve)", type=["png", "jpg", "jpeg"])
prompt = st.text_input("Enter a prompt", "a futuristic cityscape")

if uploaded_image and uploaded_mask and prompt:
    try:
        # Load and process the image and mask
        input_image = Image.open(uploaded_image).convert("RGB").resize((512, 512))
        mask_image = Image.open(uploaded_mask).convert("L").resize((512, 512))

        # Convert mask to binary (white = 255, black = 0)
        mask_array = np.array(mask_image)
        mask_array = (mask_array > 128).astype(np.uint8) * 255
        mask_image = Image.fromarray(mask_array)

        # Display inputs
        st.image(input_image, caption="Input Image", use_column_width=True)
        st.image(mask_image, caption="Mask", use_column_width=True)

        # Generate the output image
        with st.spinner("Generating image... (this may take a few minutes on CPU)"):
            output_image = pipe(
                prompt=prompt,
                image=input_image,  # Base image
                mask_image=mask_image,  # Mask for inpainting
                control_image=input_image,  # ControlNet uses the original image as guidance
                height=512,
                width=512,
                num_inference_steps=20,
                guidance_scale=3.5,
                controlnet_conditioning_scale=1.0
            ).images[0]

        # Display the result
        st.image(output_image, caption="Generated Image", use_column_width=True)

        # Provide download option
        buf = io.BytesIO()
        output_image.save(buf, format="PNG")
        byte_im = buf.getvalue()
        st.download_button(
            label="Download Generated Image",
            data=byte_im,
            file_name="generated_image.png",
            mime="image/png"
        )

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
else:
    st.write("Please upload both an image and a mask, and enter a prompt to generate an image.")
