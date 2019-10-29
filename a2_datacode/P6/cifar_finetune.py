'''
This is starter code for Assignment 2 Problem 6 of CMPT 726 Fall 2019.
The file is adapted from the repo https://github.com/chenyaofo/CIFAR-pretrained-models
'''

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
NUM_EPOCH = 1

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class CifarResNet(nn.Module):

    def __init__(self, block, layers, num_classes=100):
        super(CifarResNet, self).__init__()
        self.inplanes = 16
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

######################################################
# ##### Do not modify the code above this line ##### #
######################################################

from prettytable import PrettyTable
import pickle
import webbrowser
from PIL import Image
import numpy as np
import traceback


def reconstruct_images(rgb, filenames):
    images = []
    for i, data in enumerate(rgb):
        r = data[0:1024]
        g = data[1024:2048]
        b = data[2048:3072]
        image_array = np.array([r, g, b])  # zip(r, g, b)
        image_array = image_array.T.reshape(32, 32, 3)
        # Convert the pixels into an array using numpy
        # array = np.array(pixels, dtype=np.uint8)

        # Use PIL to create an image from the new array of pixels
        image = Image.fromarray(image_array, mode='RGB')
        image.save("./img/"+filenames[i].decode("utf-8"))

        images.append(image)
    return images


def get_meta():
    with open("D:\\00_SFU\\00_Graduate_Courses\\00_CMPT726_ML\\Assignments\\2\\a2_datacode\\P6\\data\\cifar-10-batches-py\\batches.meta",'rb') as fo:
        meta_data = pickle.load(fo)
    return {i: label for i, label in enumerate(meta_data['label_names'])}


def check_test_loss(loader):
    test_criterion = nn.CrossEntropyLoss()
    running_test_loss = 0.0
    all_test_outputs = []
    all_test_labels = []
    all_test_images = []
    import matplotlib.pyplot as plt
    for index, test_data in enumerate(loader, 0):
        test_inputs, test_labels = test_data
        test_outputs = model(test_inputs)
        test_loss = test_criterion(test_outputs, test_labels)
        running_test_loss += test_loss.item()

        all_test_outputs.extend(np.argmax(test_outputs.detach().numpy(), axis=1))
        all_test_labels.extend(test_labels.detach().numpy())

    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(all_test_labels, all_test_outputs) * 100
    print("2.1 TEST ACCURACY: "+str(accuracy)+" %")
    return running_test_loss / 10000


def test(model, loader, url=None, chrome_path=None):

    meta_data = get_meta()
    test_outputs = []
    p = PrettyTable()
    p.field_names = ["Image", "Class 0", "Class 1", "Class 2", "Class 3", "Class 4", "Class 5", "Class 6", "Class 7",
                     "Class 8", "Class 9", "Act_Index", "Act_Label", "Pred_Index",  "Pred_Label", "Associated File name"]
    p.align = "l"
    test_file = "D:\\00_SFU\\00_Graduate_Courses\\00_CMPT726_ML\\Assignments\\2" + \
                "\\a2_datacode\\P6\\data\\cifar-10-batches-py\\test_batch"

    for i, data in enumerate(loader, 0):
        # get the inputs
        test_inputs, test_labels = data
        # test_inputs = np.vstack(test_inputs).reshape(-1, 3, 32, 32)
        # test_inputs = test_inputs.transpose((0, 2, 3, 1))
        predictions = model(test_inputs)
        test_outputs.extend(predictions.detach().numpy())

    with open(test_file, 'rb') as fo:
        image_data = pickle.load(fo, encoding='bytes')
        filenames = image_data[b'filenames']
        images = reconstruct_images(image_data[b'data'], filenames)
        test_labels = image_data[b'labels']

        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(test_labels, np.argmax(test_outputs, axis=1)) * 100
        print("1 TEST ACCURACY: " + str(accuracy) + " %")

        for i, file in enumerate(filenames):
            try:
                # Get the indices of maximum element in numpy array
                max_index = np.where(test_outputs[i] == np.amax(test_outputs[i]))
                if len(max_index) > 1:
                    print('Multiple labels found', file, max_index)

                img_tag = "<img src=\"./img/" + file.decode("utf-8") + "\">"
                p.add_row([img_tag
                              , round(test_outputs[i][0], 5)
                              , round(test_outputs[i][1], 5)
                              , round(test_outputs[i][2], 5)
                              , round(test_outputs[i][3], 5)
                              , round(test_outputs[i][4], 5)
                              , round(test_outputs[i][5], 5)
                              , round(test_outputs[i][6], 5)
                              , round(test_outputs[i][7], 5)
                              , round(test_outputs[i][8], 5)
                              , round(test_outputs[i][9], 5)
                              , test_labels[i], meta_data[test_labels[i]]
                              , max_index[0][0], meta_data[max_index[0][0]]
                              , file.decode("utf-8")])
            except Exception as e:
                print(traceback.print_exc())
                break

        p.title = "Classification report over Testing dataset using ResNet20 @ Test Accuracy of 68.58%"
        html_code = p.get_html_string().replace('&lt;', '<').replace('&gt;', '>').replace('&quot;', '"')
        test_html = "D:\\00_SFU\\00_Graduate_Courses\\00_CMPT726_ML\\Assignments\\2\\a2_datacode\\P6\\test.html"
        with open(test_html, "w") as html:
            html.write(html_code)

        if url is not None:
            new = 2  # open in a new tab, if possible
            webbrowser.get(chrome_path).open(url, new=new)


class cifar_resnet20(nn.Module):
    def __init__(self):
        super(cifar_resnet20, self).__init__()
        ResNet20 = CifarResNet(BasicBlock, [3, 3, 3])
        url = 'https://github.com/chenyaofo/CIFAR-pretrained-models/releases/download/resnet/cifar100-resnet20-8412cc70.pth'
        ResNet20.load_state_dict(model_zoo.load_url(url))
        modules = list(ResNet20.children())[:-1]

        # Turning on training for ResNet20's layers
        # for param in ResNet20.parameters():
            # param.requires_grad = False  # turn all gradient off
            # print(param.requires_grad)

        backbone = nn.Sequential(*modules)
        self.backbone = nn.Sequential(*modules)
        self.fc = nn.Linear(64, 10)
        #self.fc1 = nn.Linear(64, 32)
        #self.fc2 = nn.Linear(32, 16)
        #self.fc3 = nn.Linear(16, 10)
        #self.softmax = nn.Softmax()

    def forward(self, x):
        out = self.backbone(x)
        out = out.view(out.shape[0], -1)
        #out = self.fc1(out)
        #out = self.fc2(out)
        #out = self.fc3(out)
        #return self.softmax(out)
        return self.fc(out)


NUM_EPOCH = 10

if __name__ == '__main__':
    PATH = ".\\cifar_trained.pth"

    batch_size = 100
    model = cifar_resnet20()
    #model.load_state_dict(torch.load(PATH))
    #model.eval()

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                                         std=(0.2023, 0.1994, 0.2010))])
    trainset = datasets.CIFAR10('./data', download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    test_set = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(list(model.fc.parameters()), lr=0.001, momentum=0.9, weight_decay=0.001)
    # optimizer = optim.SGD(dict({model.fc.parameters(), model.fc2.parameters(), model.fc3.parameters()}), lr=0.001, momentum=0.9, weight_decay=0.001)

    # Do the training
    for epoch in range(NUM_EPOCH):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 50 == 49:    # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f ' % (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0
        # print('[ EPOCH = %d ]' % (epoch + 1), "Test Loss: ", check_test_loss(test_loader))

    print('Finished Training')

    # model = cifar_resnet20()
    # model.load_state_dict(torch.load(PATH))
    # model.eval()

    # Final Testing
    url = "D:/00_SFU/00_Graduate_Courses/00_CMPT726_ML/Assignments/2/a2_datacode/P6/test.html"
    chrome_path = 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s'
    check_test_loss(test_loader, test_set)
    # test(model, test_loader, url, chrome_path)

    #torch.save(model.state_dict(), PATH)




