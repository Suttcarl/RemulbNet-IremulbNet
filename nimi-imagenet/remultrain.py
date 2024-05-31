import os
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from Remulbnet import remulbnet
import matplotlib.pyplot as plt
import time
from PIL import Image
import re
from thop import profile
from thop import clever_format
from my_dataset import MyDataSet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.synchronize()
    startg = time.time()
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.Resize(256),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   #transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    #cwd = os.getcwd()
    #data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    json_path = "./classes_name.json"
    image_path = os.path.join("/home/sutiantian/miniimagenet/data/")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = MyDataSet(root_dir=image_path,
                              csv_name="new_train.csv",
                              json_path=json_path,
                              transform=data_transform["train"])
    train_num = len(train_dataset)


    '''flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)'''

    batch_size = 128
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    val_dataset = MyDataSet(root_dir=image_path,
                            csv_name="new_val.csv",
                            json_path=json_path,
                            transform=data_transform["val"])
    val_num = len(val_dataset)
    validate_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    net = remulbnet(num_classes=100).to(device)
    '''model_weight_path = "./resnet34-333f7ec4.pth"#迁移学习
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 17)
    net.to(device)'''
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    '''params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)'''
    
    
    
    epochs = 200
    best_acc = 0.0
    test_acc=0.0
    a=0
    save_path = './remulbnet.pth'
    train_steps = len(train_loader)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    plt.title("training loss and val_accurate")
    ax.set_xlabel("epoch")
    ax.set_ylabel("")
    y1=[]#loss
    y2=[]#val
    x=[]#epoch


    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)


        val_accurate = acc / val_num
        y1.append(running_loss / train_steps)
        y2.append(val_accurate)
        x.append(epoch+1)
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f  ' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
            a=epoch

    
            
    line1, = ax.plot(x, y1, 'b-', label='training loss')
    line2, = ax.plot(x, y2, 'r-', label='val_accurate')
    print('epoch:  %d   val_acc:  %.3f   '%(a+1,y2[a]))
    torch.cuda.synchronize()
    endg = time.time()
    print('training time: %.3f ' % (endg-startg))
    
    fil=open("remulbnet.txt","w")
    y=[float('{:.3f}'.format(i)) for i in y1]
    fil.writelines(str(y)+str('\r'))
    y=[float('{:.3f}'.format(i)) for i in y2]
    fil.writelines(str(y)+str('\r'))
    fil.close()
    
    '''total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameter: %.2fM" % (total/1e6))'''
    model = remulbnet()
    input = torch.randn(1,3,224,224)
    flops,params = profile(model,inputs=(input, ))
    flops,params = clever_format([flops,params],"%.3f")
    print("FLOPS: %s PARAMS: %s  "%(flops,params))
    #os.system("python ibatch_predict.py")
    print('Finished Training')
    ax.legend(handles=[line1, line2], labels=['training loss', 'val_accurate'], loc='best')
    plt.savefig('lossandacc.jpg')
    plt.pause(3)
    plt.close(fig)
    
if __name__ == '__main__':
    main()
