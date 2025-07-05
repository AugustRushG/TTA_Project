import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import torch

# load your image
img = Image.open("/home/august/github/TTA_Project/data/24Paralympics_FRA_M9_R16_Mai_UKR_v_Leibovitz_USA_game_1_frames/000571.jpg").convert("RGB")

# define the center crop transform
center_crop = transforms.CenterCrop(size=(224, 224))

# apply the transform
cropped_img = center_crop(img)

# plot original and cropped side by side
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(img)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Center Cropped")
plt.imshow(cropped_img)
plt.axis("off")

plt.show()