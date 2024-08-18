from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline
import torch


def set_model():
    model_id = "timbrooks/instruct-pix2pix"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id)
    pipe = pipe.to(device)
    return pipe


def get_prompt(body_fat):
    if body_fat <= 5:
        return ("Transform the image to depict a realistic improvement in physique, representing a person who has achieved a healthy 5% body fat after a month of dedicated training. The abdominal muscles should be visible and defined, but maintain a natural softness around the edges, indicative of a fit, but not overly muscular build. Focus on enhancing the overall muscle tone, creating a balanced and athletic appearance, with a subtle emphasis on increased muscularity and improved body shape.")

    if body_fat <= 10:
        return ("Transform the image to depict a realistic improvement in physique, representing a person who has achieved a healthy 10% body fat after a month of dedicated training. The abdominal muscles should be visible and defined, but maintain a natural softness around the edges, indicative of a fit, but not overly muscular build. Focus on enhancing the overall muscle tone, creating a balanced and athletic appearance, with a subtle emphasis on increased muscularity and improved body shape.")

    if body_fat <= 15:
        return ("Transform the image to depict a realistic improvement in physique, representing a person who has achieved a healthy 15% body fat after a month of dedicated training. The abdominal muscles should be visible and defined, but maintain a natural softness around the edges, indicative of a fit, but not overly muscular build. Focus on enhancing the overall muscle tone, creating a balanced and athletic appearance, with a subtle emphasis on increased muscularity and improved body shape.")


def generate_image(input_image_path, body_fat):
    input_image = Image.open(input_image_path).convert("RGB")
    prompt = get_prompt(body_fat)
    # Apply the model to the input image with the given prompt
    generated_image = pipe(
        prompt=prompt, image=input_image, num_inference_steps=100
    ).images[0]
    generated_image.save("generated_image.png")


## Image path
input_image_path = "path for image"
body_fat = 10
generate_image(input_image_path, body_fat)