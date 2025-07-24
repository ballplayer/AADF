# import torch
# import torchvision
# import torchvision.transforms as transforms
# from torch import nn
# from torch.utils.data import DataLoader
# import numpy as np
#
#
# batch_size = 128
# clip_values = (0, 1)
# nb_classes = 10
# shape = (1, 28, 28)
# preprocessing = (np.array([0.1307]), np.array([0.3081]))
# transform = transforms.Compose([transforms.Grayscale(1),
#                                 transforms.ToTensor(),
#                                 transforms.Normalize(*preprocessing)
#                                 ])
# train_dataset = torchvision.datasets.MNIST(root='../data', train=True, transform=transform, download=True)
# test_dataset = torchvision.datasets.MNIST(root='../data', train=False, transform=transform, download=True)
# train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
# test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
#
#
# model = torchvision.models.resnet50()
# model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
# model.load_state_dict(torch.load('../model-weights/resnet50-single.pth'))
# model.fc = nn.Sequential(
#             model.fc,
#             nn.ReLU(inplace=False),
#             nn.Dropout(0.5, inplace=False),
#             nn.Linear(1000, nb_classes)
#         )
#
# for param in model.parameters():
#     param.requires_grad = False
# for param in model.fc.parameters():
#     param.requires_grad = True
#
#
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# loss = nn.CrossEntropyLoss()
# epochs = 50
# model.to('cuda')
# loss.to('cuda')
# for epoch in range(epochs):
#     model.train()
#     total_train_loss = 0
#     total_acc = 0
#     for images, labels in train_data_loader:
#         images = images.to('cuda')
#         labels = labels.to('cuda')
#         outputs = model(images)
#         l = loss(outputs, labels)
#         total_train_loss += l.item()
#         total_acc += (outputs.argmax(dim=1) == labels).sum().item()
#         optimizer.zero_grad()
#         l.backward()
#         optimizer.step()
#     print('epoch: {}, Loss: {:.4f}, Accuracy: {:.4f}'.format(epoch + 1, total_train_loss / len(train_dataset), total_acc / len(train_dataset)))
#     torch.save(model.state_dict(), '../model-weights/resnet50-mnist-{:.4f}.pth'.format(total_acc / len(train_dataset)))
#
#     model.eval()
#     with torch.no_grad():
#         accuracy = 0
#         for images, labels in test_data_loader:
#             images = images.to('cuda')
#             labels = labels.to('cuda')
#
#             output = model(images)
#             accuracy += (output.argmax(dim=1) == labels).sum()
#     print("epoch:{}, 测试集上的准确率为: {:.4f}".format(epoch + 1, accuracy / len(test_dataset)))
