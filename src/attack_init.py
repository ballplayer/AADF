import foolbox as fb
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from art.utils import to_categorical
from torch.utils.data import DataLoader
import numpy as np
import torch

from art.estimators.classification import PyTorchClassifier

from foolbox.distances import LpDistance


def dataset_init(dataset_name, model_name, method_name, targeted, target_class):
    global dataset, data_loader, preprocessing, shape, clip_values, nb_classes, y_target

    batch_size = 1
    shuffle = False
    if method_name == 'DatasetAttack':
        batch_size = 8
    if dataset_name == 'MSTAR' and model_name in ['ResNet50', 'VGG16', 'GoogleNet']:
        clip_values = (0, 1)
        nb_classes = 8
        shape = (3, 224, 224)
        preprocessing = (np.array([0.2531, 0.2531, 0.2531]), np.array([0.2093, 0.2093, 0.2093]))

        transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor()
                                        ])
        dataset = torchvision.datasets.ImageFolder(root='../data/MSTAR/test50', transform=transform)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    elif dataset_name == 'ImageNet' and model_name in ['ResNet50', 'VGG16', 'GoogleNet']:
        clip_values = (0, 1)
        nb_classes = 1000
        shape = (3, 224, 224)
        preprocessing = (np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225]))
        transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor()
                                        ])
        dataset = torchvision.datasets.ImageFolder(root='../data/ImageNet/test50', transform=transform)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    elif dataset_name == 'MNIST' and model_name in ['ResNet50', 'VGG16', 'GoogleNet']:
        clip_values = (0, 1)
        nb_classes = 10
        shape = (1, 64, 64)
        preprocessing = (np.array([0.1307]), np.array([0.3081]))
        transform = transforms.Compose([transforms.Resize(64),
                                        transforms.ToTensor()
                                        ])
        dataset = torchvision.datasets.MNIST(root='../data', train=False, transform=transform, download=True)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    elif dataset_name == 'CIFAR10' and model_name in ['ResNet50', 'VGG16', 'GoogleNet']:
        clip_values = (0, 1)
        nb_classes = 10
        shape = (3, 32, 32)
        preprocessing = (np.array([0.4914, 0.4822, 0.4465]), np.array([0.2471, 0.2435, 0.2616]))
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = torchvision.datasets.CIFAR10(root='../data', train=False, transform=transform, download=False)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    else:
        print("所选择的数据集没有对应的模型！")
        exit(0)

    if method_name in ['AutoAttack', 'GeoDA', 'SquareAttack']:
        target_class = 0
    elif method_name in ['AutoConjugateGradient', 'AutoProjectedGradientDescent', 'SaliencyMapMethod']:
        target_class = 1

    if targeted and target_class is not None:
        y_target = to_categorical([target_class for _ in range(batch_size)], nb_classes)
    else:
        y_target = None

    return data_loader, batch_size, shape, clip_values, preprocessing, nb_classes, y_target


def model_init(model_name, dataset_name, nb_classes, device):
    global model

    if model_name == 'ResNet50' and dataset_name == 'MSTAR':
        model = torchvision.models.resnet50()
        # 修改（2048，1000），添加（1000，8）
        model.fc = nn.Sequential(
            nn.Linear(2048, 1000),
            nn.ReLU(inplace=False),
            nn.Linear(1000, nb_classes)
        )
        model.load_state_dict(
            torch.load("../model-weights/MSTAR-resnet50-0.9984.pth", map_location=torch.device('cpu')))
    elif model_name == 'VGG16' and dataset_name == 'MSTAR':
        model = torchvision.models.vgg16()
        # 修改（4096，1000），添加（1000，8）
        model.classifier[6] = nn.Sequential(
            nn.Linear(4096, 1000),
            nn.ReLU(inplace=False),
            nn.Linear(1000, nb_classes)
        )
        model.load_state_dict(torch.load("../model-weights/MSTAR-vgg16-0.9937.pth", map_location=torch.device('cpu')))
    elif model_name == 'GoogleNet' and dataset_name == 'MSTAR':
        model = torchvision.models.googlenet()
        # 修改（1024，1000），添加（1000，8）
        model.fc = nn.Sequential(
            nn.Linear(1024, 1000),
            nn.ReLU(inplace=False),
            nn.Linear(1000, nb_classes)
        )
        model.load_state_dict(
            torch.load("../model-weights/MSTAR-googlenet-0.9989.pth", map_location=torch.device('cpu')))
    elif dataset_name == 'ImageNet' and model_name == "ResNet50":
        model = torchvision.models.resnet50()
        model.load_state_dict(torch.load("../model-weights/resnet50-11ad3fa6.pth", map_location=torch.device('cpu')))
    elif dataset_name == 'ImageNet' and model_name == "VGG16":
        model = torchvision.models.vgg16()
        model.load_state_dict(torch.load("../model-weights/vgg16-397923af.pth", map_location=torch.device('cpu')))
    elif dataset_name == 'ImageNet' and model_name == "GoogleNet":
        model = torchvision.models.googlenet()
        model.load_state_dict(torch.load("../model-weights/googlenet-1378be20.pth", map_location=torch.device('cpu')))
    elif dataset_name == 'MNIST' and model_name == "ResNet50":
        model = torchvision.models.resnet50()
        # 单通道
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 修改（2048，1000），添加（1000，10）
        model.fc = nn.Sequential(
            nn.Linear(2048, 1000),
            nn.ReLU(inplace=False),
            nn.Linear(1000, nb_classes)
        )
        model.load_state_dict(
            torch.load("../model-weights/MNIST-resnet50-0.9935.pth", map_location=torch.device('cpu')))
    elif dataset_name == 'MNIST' and model_name == "VGG16":
        model = torchvision.models.vgg16()
        # 单通道
        model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # 修改（4096，1000），添加（1000，10）
        model.classifier[6] = nn.Sequential(
            nn.Linear(4096, 1000),
            nn.ReLU(inplace=False),
            nn.Linear(1000, nb_classes)
        )
        model.load_state_dict(torch.load("../model-weights/MNIST-vgg16-9922.pth", map_location=torch.device('cpu')))
    elif dataset_name == 'MNIST' and model_name == "GoogleNet":
        model = torchvision.models.googlenet()
        # 单通道
        model.conv1.conv = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # 修改（1024，1000），添加（1000，10）
        model.fc = nn.Sequential(
            nn.Linear(1024, 1000),
            nn.ReLU(inplace=False),
            nn.Linear(1000, nb_classes)
        )
        model.load_state_dict(
            torch.load("../model-weights/MNIST-googlenet-0.9937.pth", map_location=torch.device('cpu')))
    elif dataset_name == 'CIFAR10' and model_name == "ResNet50":
        model = torchvision.models.resnet50()
        # 修改（2048，1000），添加（1000，10）
        model.fc = nn.Sequential(
            nn.Linear(2048, 1000),
            nn.ReLU(inplace=False),
            nn.Linear(1000, nb_classes)
        )
        model.load_state_dict(
            torch.load("../model-weights/CIFAR10-resnet50-0.8918.pth", map_location=torch.device('cpu')))
    elif dataset_name == 'CIFAR10' and model_name == "VGG16":
        model = torchvision.models.vgg16()
        model.classifier[6] = nn.Sequential(
            nn.Linear(4096, 1000),
            nn.ReLU(inplace=False),
            nn.Linear(1000, nb_classes)
        )
        model.load_state_dict(torch.load("../model-weights/CIFAR10-vgg16-0.9083.pth", map_location=torch.device('cpu')))
    elif dataset_name == 'CIFAR10' and model_name == "GoogleNet":
        model = torchvision.models.googlenet()
        model.fc = nn.Sequential(
            nn.Linear(1024, 1000),
            nn.ReLU(inplace=False),
            nn.Linear(1000, nb_classes)
        )
        model.load_state_dict(
            torch.load("../model-weights/CIFAR10-googlenet-0.8668.pth", map_location=torch.device('cpu')))
    else:
        print("所选择的数据集没有对应的模型！")
        exit(0)

    model.to(device)
    model.eval()
    return model


def fb_init(model, clip_values, method_name, preprocessing, steps, distance_str, device):
    global ae_model, attack, distance

    ae_model = fb.PyTorchModel(model, bounds=clip_values, device=device,
                               preprocessing=dict(mean=list(map(float, preprocessing[0])), std=list(map(float, preprocessing[1])), axis=-3)).transform_bounds(clip_values)
    if distance_str is not None:
        distance = (LpDistance(float(distance_str[1:])) if distance_str == 'linf' else LpDistance(int(distance_str[1:])))

    if method_name == 'VirtualAdversarialAttack':
        attack = fb.attacks.VirtualAdversarialAttack(steps=steps)
    elif method_name == 'BinarySearchContrastReductionAttack':
        attack = fb.attacks.BinarySearchContrastReductionAttack(distance=distance)
    elif method_name == 'BinarizationRefinementAttack':
        attack = fb.attacks.BinarizationRefinementAttack(distance=distance)
    elif method_name == 'DatasetAttack':
        attack = fb.attacks.DatasetAttack(distance=distance)
    elif method_name == 'GaussianBlurAttack':
        steps = (1000 if steps is None else steps)
        attack = fb.attacks.GaussianBlurAttack(distance=distance, steps=steps)
    elif method_name == 'LinearSearchBlendedUniformNoiseAttack':
        steps = (1000 if steps is None else steps)
        attack = fb.attacks.LinearSearchBlendedUniformNoiseAttack(distance=distance, steps=steps)
    elif method_name == 'LinearSearchContrastReductionAttack':
        steps = (1000 if steps is None else steps)
        attack = fb.attacks.LinearSearchContrastReductionAttack(distance=distance, steps=steps)
    elif method_name == 'BoundaryAttack':
        steps = (25000 if steps is None else steps)
        attack = fb.attacks.BoundaryAttack(steps=steps)
    elif method_name == 'L2CarliniWagnerAttack':
        steps = (10000 if steps is None else steps)
        attack = fb.attacks.L2CarliniWagnerAttack(steps=steps)
    elif method_name == 'DDNAttack':
        attack = fb.attacks.DDNAttack()
    elif method_name == 'EADAttack':
        steps = (10000 if steps is None else steps)
        attack = fb.attacks.EADAttack(steps=steps)
    elif method_name in ['FGSM', 'LinfFastGradientAttack']:
        attack = fb.attacks.LinfFastGradientAttack()
    elif method_name == 'NewtonFoolAttack':
        attack = fb.attacks.NewtonFoolAttack()
    elif method_name in ['PGD', 'LinfPGD', 'LinfProjectedGradientDescentAttack']:
        attack = fb.attacks.LinfProjectedGradientDescentAttack()
    elif method_name == 'SaltAndPepperNoiseAttack':
        steps = (1000 if steps is None else steps)
        attack = fb.attacks.SaltAndPepperNoiseAttack(steps=steps)
    elif method_name == 'L1BrendelBethgeAttack':
        steps = (1000 if steps is None else steps)
        attack = fb.attacks.L1BrendelBethgeAttack(steps=steps)
    elif method_name == 'L1FMNAttack':
        attack = fb.attacks.L1FMNAttack()
    elif method_name == 'L2AdditiveGaussianNoiseAttack':
        attack = fb.attacks.L2AdditiveGaussianNoiseAttack()
    elif method_name == 'L2AdditiveUniformNoiseAttack':
        attack = fb.attacks.L2AdditiveUniformNoiseAttack()
    elif method_name == 'L2BasicIterativeAttack':
        attack = fb.attacks.L2BasicIterativeAttack()
    elif method_name == 'L2BrendelBethgeAttack':
        steps = (1000 if steps is None else steps)
        attack = fb.attacks.L2BrendelBethgeAttack(steps=steps)
    elif method_name == 'L2ClippingAwareAdditiveGaussianNoiseAttack':
        attack = fb.attacks.L2ClippingAwareAdditiveGaussianNoiseAttack()
    elif method_name == 'L2ClippingAwareAdditiveUniformNoiseAttack':
        attack = fb.attacks.L2ClippingAwareAdditiveUniformNoiseAttack()
    elif method_name == 'L2ClippingAwareRepeatedAdditiveGaussianNoiseAttack':
        attack = fb.attacks.L2ClippingAwareRepeatedAdditiveGaussianNoiseAttack()
    elif method_name == 'L2ClippingAwareRepeatedAdditiveUniformNoiseAttack':
        attack = fb.attacks.L2ClippingAwareRepeatedAdditiveUniformNoiseAttack()
    elif method_name == 'L2ContrastReductionAttack':
        attack = fb.attacks.L2ContrastReductionAttack()
    elif method_name == 'L2DeepFoolAttack':
        attack = fb.attacks.L2DeepFoolAttack()
    elif method_name in ['FGM', 'L2FastGradientAttack']:
        attack = fb.attacks.L2FastGradientAttack()
    elif method_name == 'L2FMNAttack':
        attack = fb.attacks.L2FMNAttack()
    elif method_name in ['L2PGD', 'L2ProjectedGradientDescentAttack']:
        attack = fb.attacks.L2ProjectedGradientDescentAttack()
    elif method_name == 'L2RepeatedAdditiveGaussianNoiseAttack':
        attack = fb.attacks.L2RepeatedAdditiveGaussianNoiseAttack()
    elif method_name == 'L2RepeatedAdditiveUniformNoiseAttack':
        attack = fb.attacks.L2RepeatedAdditiveUniformNoiseAttack()
    elif method_name == 'LinfAdditiveUniformNoiseAttack':
        attack = fb.attacks.LinfAdditiveUniformNoiseAttack()
    elif method_name == 'LinfBasicIterativeAttack':
        attack = fb.attacks.LinfBasicIterativeAttack()
    elif method_name == 'LinfDeepFoolAttack':
        attack = fb.attacks.LinfDeepFoolAttack()
    elif method_name == 'LInfFMNAttack':
        attack = fb.attacks.LInfFMNAttack()
    elif method_name == 'LinfinityBrendelBethgeAttack':
        steps = (1000 if steps is None else steps)
        attack = fb.attacks.LinfinityBrendelBethgeAttack(steps=steps)
    elif method_name == 'LinfRepeatedAdditiveUniformNoiseAttack':
        attack = fb.attacks.LinfRepeatedAdditiveUniformNoiseAttack()
    return ae_model, attack


def art_init(model, nb_classes, method_name, shape, clip_values, preprocessing, device):
    global ae_model
    if method_name in ['AdversarialPatch', 'AutoAttack', 'CarliniL0Method', 'CarliniLInfMethod', 'GeoDA',
                       'SquareAttack']:
        ae_model = PyTorchClassifier(model=model, clip_values=clip_values, loss=None, input_shape=shape,
                                     nb_classes=nb_classes, preprocessing=preprocessing, device_type=device)
    elif method_name in ['AutoProjectedGradientDescent', 'AutoConjugateGradient', 'SaliencyMapMethod']:
        ae_model = PyTorchClassifier(model=model, clip_values=clip_values, loss=nn.CrossEntropyLoss(), input_shape=shape,
                                     nb_classes=nb_classes,  preprocessing=preprocessing, device_type=device)

    return ae_model