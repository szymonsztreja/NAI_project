import torch
from Flickr8kDataSet import Flickr8kDataSet
from diffusers import StableDiffusionPipeline, DiffusionPipeline, AutoPipelineForText2Image
from torchmetrics.functional.multimodal import clip_score
from functools import partial
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms import functional as F
from torchmetrics.image.fid import FrechetInceptionDistance


# model_id = "stabilityai/stable-diffusion-2-1"

flickr_dataset = Flickr8kDataSet("flickr8k/single_captions.csv","flickr8k/Images/")

prompts = flickr_dataset.img_labels[:10]['caption']
prompts = prompts.tolist()


clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

def calculate_clip_score(images, prompts):
    images_int = (images * 255).astype("uint8")
    clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
    return round(float(clip_score), 4)


def preprocess_image(image):
    image = image.unsqueeze(0)
    return F.center_crop(image, (300, 300))


real_images = []

for i in range(10):
    image, label = flickr_dataset[i]
    real_images.append(image)
    print(image.shape)


real_images = torch.cat([preprocess_image(image) for image in real_images])

sd_2_1_pipeline = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", image_resolution=(300, 300), torch_dtype=torch.float16).to("cuda")
sd_2_1_images = sd_2_1_pipeline(prompts, num_images_per_prompt=1, output_type="np").images
sd_2_1_clip_score = calculate_clip_score(sd_2_1_images, prompts)




fake_images = torch.tensor(sd_2_1_images)
fake_images = fake_images.permute(0, 3, 1, 2)

fid = FrechetInceptionDistance(normalize=True)
fid.update(real_images, real=True)
fid.update(fake_images, real=False)

print(f"sd_2_1 CLIP score: {sd_2_1_clip_score}")
print(f"FID: {float(fid.compute())}")

