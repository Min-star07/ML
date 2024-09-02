from torch.utils.tensorboard import SummaryWriter
import cv2

writer = SummaryWriter("logs")

for i in range(100):
    writer.add_scalar("y=2*x", 2 * i, i)

writer.close()


img_path = "./data/hymenoptera_data/train/ants/0013035.jpg"
img = cv2.imread(img_path)
print(type(img))
print(img.shape)
writer.add_image("my_image", img, dataformats="HWC")
writer.close()
