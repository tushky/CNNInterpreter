import sys
import os
import torchvision
sys.path.insert(1, './utils')
from utils import read_image, postprocess
import matplotlib.pyplot as plt
from ClassActivationMaps import ClassActivationMaps
from tqdm import tqdm


model = torchvision.models.resnet34(pretrained=True)
cam = ClassActivationMaps(model)
cam.interactive = True
method = 'gradcam++'
for i in tqdm(range(132)):
    file_name = f'frame{i}.jpg'
    image = read_image(os.getcwd()+'/test/frames/'+file_name)
    plt.gcf().set_size_inches(8, 8)
    plt.axis('off')
    out = cam.get_cam(image, method, class_index=236)
    img = postprocess(image)
    # plot input image
    plt.imshow(out, cmap='jet')
    plt.imshow(img, alpha=0.5)
    # save overlapping output
    plt.savefig(f'./data/frames/frames{i}.png', bbox_inches='tight')
    plt.cla()
    #break