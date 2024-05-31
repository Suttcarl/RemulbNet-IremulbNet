import os
import json
import re
import torch
from PIL import Image
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from tqdm import tqdm
from my_dataset import MyDataSet
from thop import profile
from thop import clever_format


from resnet import resnet50  
from dmodel import densenet121  
from model_v2 import MobileNetV2 
from model_v3 import mobilenet_v3_large  
from smodel import shufflenet_v2_x1_0   
from rgmodel import create_regnet  
from PeleeNet import PeleeNet 
from inceptionnext import inceptionnext_tiny
from Iremulbnet import iremulbnet as model0

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [#transforms.Resize([380,300]),
         transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    #cwd = os.getcwd()
    json_path = "./classes_name.json"
    image_path = os.path.join("/home/sutiantian/miniimagenet/data/")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    
    
    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    val_dataset = MyDataSet(root_dir=image_path,
                            csv_name="new_val.csv",
                            json_path=json_path,
                            transform=data_transform)
    val_num = len(val_dataset)
    validate_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print(val_num)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = model0(num_classes=100).to(device)
    weights_path = "./iremulbnet.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))
  
    tn=0
    fn=0
    
    acc = 0.0 
    # prediction
    model.eval()
    with torch.no_grad():
        # predict class
        val_bar = tqdm(validate_loader)
        for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = model(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
        val_accurate = acc / val_num
    print(val_accurate)
    
    model = model0()
    input = torch.randn(1,3,224,224)
    flops,params = profile(model,inputs=(input, ))
    flops,params = clever_format([flops,params],"%.3f")
    print("FLOPS: %s PARAMS: %s  "%(flops,params))
if __name__ == '__main__':
    main()
    
