import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# ===================== 1. 模型定义 =====================
class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 1,out_channels = 16,kernel_size = 3,stride = 1,padding = 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2,stride = 2),
            torch.nn.Conv2d(in_channels = 16,out_channels = 32,kernel_size = 3,stride = 1,padding = 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2,stride = 2),
            torch.nn.Conv2d(in_channels = 32,out_channels = 64,kernel_size = 3,stride = 1,padding = 1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features = 7 * 7 * 64,out_features = 128),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features = 128,out_features = 10),
        )
    def forward(self,input):
        output = self.model(input)
        return output

# ===================== 2. 加载模型 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Net().to(device)
net.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
net.eval()
print("模型加载完成！")

# ===================== 3. 图片预处理 =====================
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(28),
    transforms.CenterCrop(28),
    transforms.ToTensor(),
])

# ===================== 4. 读取图片、反转+二值化 =====================
img_path = r"E:\VS Code Projects\pytorch\num6.png"
img = Image.open(img_path)
img_tensor = transform(img).unsqueeze(0)
img_tensor = 1.0 - img_tensor

# 自适应阈值：比背景均值高 0.15 才算笔画
bg = img_tensor.mean()
threshold = bg + 0.15
img_tensor = (img_tensor > threshold).float()

img_tensor = img_tensor.to(device)

# 推理
with torch.no_grad():
    pred_logits = net(img_tensor)
    pred = torch.argmax(pred_logits, dim=1)
    prob = torch.softmax(pred_logits, dim=1)

print(f"预测数字：{pred.item()}")
print("各类别概率：")
for i in range(10):
    print(f"数字{i}: {prob[0,i].item():.4f}")

# 可视化送入模型的图片
show_img = img_tensor[0,0].cpu()
plt.imshow(show_img, cmap="gray")
plt.title(f"Model Input, Predict = {pred.item()}")
plt.show()
