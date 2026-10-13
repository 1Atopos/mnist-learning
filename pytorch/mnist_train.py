import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

# ====================== 1. 设置设备 GPU/CPU ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备：", device)

# ====================== 2. 加载MNIST数据集，构造DataLoader ======================
BATCH_SIZE = 64

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
])

# 下载加载数据集
trainData = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)
testData = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

trainDataLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)
testDataLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=BATCH_SIZE)

# ====================== 3. 构建CNN网络模型 Net ======================
class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.model = torch.nn.Sequential(
            # The size of the picture is 28x28
            torch.nn.Conv2d(in_channels = 1,out_channels = 16,kernel_size = 3,stride = 1,padding = 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2,stride = 2),

            # The size of the picture is 14x14
            torch.nn.Conv2d(in_channels = 16,out_channels = 32,kernel_size = 3,stride = 1,padding = 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2,stride = 2),

            # The size of the picture is 7x7
            torch.nn.Conv2d(in_channels = 32,out_channels = 64,kernel_size = 3,stride = 1,padding = 1),
            torch.nn.ReLU(),

            torch.nn.Flatten(),
            torch.nn.Linear(in_features = 7 * 7 * 64,out_features = 128),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features = 128,out_features = 10),
            # ❗删掉了Softmax，因为CrossEntropyLoss自带softmax
        )

    def forward(self,input):
        output = self.model(input)
        return output

# 实例化模型并放到对应设备
net = Net()
net = net.to(device)
print(net)

# ====================== 4. 损失函数 + 优化器 ======================
lossF = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters())

# ====================== 5. 训练+测试循环 ======================
EPOCHS = 10
#存储训练过程
history = {'Test Loss':[],'Test Accuracy':[]}
for epoch in range(1,EPOCHS + 1):
    processBar = tqdm(trainDataLoader,unit = 'step')
    net.train(True)
    for step,(trainImgs,labels) in enumerate(processBar):
        trainImgs = trainImgs.to(device)
        labels = labels.to(device)
        
        net.zero_grad()
        outputs = net(trainImgs)
        loss = lossF(outputs,labels)
        predictions = torch.argmax(outputs, dim = 1)
        accuracy = torch.sum(predictions == labels)/labels.shape[0]
        
        loss.backward()
        optimizer.step()
        
        # 基础进度条显示
        processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f" % 
                                   (epoch,EPOCHS,loss.item(),accuracy.item()))
        
        # ========= 只有本轮epoch最后一个batch，才跑测试集 =========
        if step == len(processBar)-1:
            correct,totalLoss = 0,0
            net.train(False)
            # 关闭梯度计算，节省显存
            with torch.no_grad():
                for testImgs,labels in testDataLoader:
                    testImgs = testImgs.to(device)
                    labels = labels.to(device)
                    outputs = net(testImgs)
                    loss = lossF(outputs,labels)
                    predictions = torch.argmax(outputs,dim = 1)
                    
                    #存储测试结果
                    totalLoss += loss
                    correct += torch.sum(predictions == labels)
            #计算总测试的平均准确率
            testAccuracy = correct/(BATCH_SIZE * len(testDataLoader))
            #计算总测试的平均Loss
            testLoss = totalLoss/len(testDataLoader)
            #保存到history
            history['Test Loss'].append(testLoss.item())
            history['Test Accuracy'].append(testAccuracy.item())
            # 更新进度条，带上测试指标
            processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f, Test Loss: %.4f, Test Acc: %.4f" %             
                            (epoch,EPOCHS,loss.item(),accuracy.item(),testLoss.item(),testAccuracy.item()))
    processBar.close()

    # 保存训练好的模型参数
torch.save(net.state_dict(), "mnist_cnn.pth")
print("模型已保存为 mnist_cnn.pth")


# 绘制测试损失曲线
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history['Test Loss'], label='Test Loss')
plt.title('Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 绘制测试准确率曲线
plt.subplot(1,2,2)
plt.plot(history['Test Accuracy'], label='Test Accuracy',color='orange')
plt.title('Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
