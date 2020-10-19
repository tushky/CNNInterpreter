import torch
from scipy.ndimage.filters import gaussian_filter1d
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

def read_image(path):
    image = Image.open(path).convert('RGB')
    return preprocess(image)

def preprocess(image):

    '''
    Accepts PIL image and convert it into [B, C, H, W] shaped tensor normalized with imagenet stats
    '''
    transform = transforms.Compose(
                [
                    transforms.Resize(224),
                    # Reshape to size 224 X 224
                    #transforms.CenterCrop(224),
                    # Convert to torch tensor
                    transforms.ToTensor(),
                    # Normalize with imagenet stats
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    ),
                    transforms.Lambda(lambda x: x.unsqueeze(0))
                ]
            )
    return transform(image)

def process_deconv_output(t):
    '''
    accepts tensor of shape [B, C, H, W] processed with imagenet stats and convert it to PIL image
    '''

    mean=torch.tensor([0.485, 0.456, 0.406])
    std=torch.tensor([0.229, 0.224, 0.225])

    transform = transforms.Compose(
        [
            transforms.Lambda(lambda x: clamp(x)),
            # Remove normalization using imagenet stats
            transforms.Lambda(lambda t : t*std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)),
            # Remove batch dimention
            transforms.Lambda(lambda t: t.squeeze(0)),
            transforms.ToPILImage(),
            transforms.Lambda(lambda img: ImageEnhance.Contrast(img).enhance(4)),
            transforms.ToTensor()
        ]
    )
    return transform(t)


def postprocess(t):

    '''
    accepts tensor of shape [B, C, H, W] processed with imagenet stats and convert it to PIL image
    '''

    mean=torch.tensor([0.485, 0.456, 0.406])
    std=torch.tensor([0.229, 0.224, 0.225])

    transform = transforms.Compose(
        [
            # Remove normalization using imagenet stats
            transforms.Lambda(lambda t : t*std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)),
            # Remove batch dimention
            #transforms.Lambda(lambda  x: (x - x.min())/(x.max() - x.min() + 1e-06)),
            transforms.Lambda(lambda t: t.squeeze(0)),
            # Convert to PIL image
            transforms.ToPILImage()
        ]
    )
    return transform(t)

def jitter(t, x, y):

    '''
    apply jitter to tensor t of shape [batch, channel, height, width] in (x, y) direction
    '''
    # jitter in x-direction
    t = torch.cat((t[..., x:], t[..., :x]), dim = 3)
    # jitter in y-direction
    t = torch.cat((t[:, :, y:, :], t[:, :, :y, :]), dim = 2)
    return t

def blur(t, sigma=1):

    '''
    apply gausian blur with sigma to the input tensor
    '''

    X_np = t.cpu().clone().numpy()
    X_np = gaussian_filter1d(X_np, sigma, axis=2)
    X_np = gaussian_filter1d(X_np, sigma, axis=3)
    t.copy_(torch.Tensor(X_np).type_as(t))
    return t

def clamp(t):

    for c in range(t.shape[1]):
        low = - mean[c] / std[c]
        high = (1 - mean[c]) / std[c]
        t.data[0, c, :].clamp_(min=low, max=high)
    return t

if __name__ == '__main__':

    t = torch.randn((1 , 1, 4, 4))
    print(t)
    t = blur(t, sigma=0.5)
    print(t)