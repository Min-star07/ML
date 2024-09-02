from torchvision import transforms
import cv2
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")

img_path = "../data/hymenoptera_data/train/ants/0013035.jpg"

img = cv2.imread(img_path)
cv_image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
# print(img)
# ToTensor
trans_tensor = transforms.ToTensor()
img_tensor = trans_tensor(cv_image_rgb)
writer.add_image("tensor_img", img_tensor)

# print(type(img_tensor))
# print(len(img_tensor))
# print(len(img_tensor[0]))
# print(len(img_tensor[1]))
# print(len(img_tensor[2]))

# Normlize
trans_norm = transforms.Normalize(
    [0.5, 0.5, 0.5],
    [0.5, 0.5, 0.5],
)
img_norm = trans_norm(img_tensor)
writer.add_image("tensor_img", img_norm)

# Resize
print(cv_image_rgb.shape)
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img_tensor)
print(img_resize.shape)
writer.close()

# Compose -resize
