import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
import torchvision.transforms as transforms
from PIL import Image
import torch
torch.set_grad_enabled(False)

weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights, progress=False)
model.eval()


## Note: may need to modify start_res (randomly added a default value)
def add_object_diffusion_noise(image_path=None, blur_strength=1, start_res=100, background_blurring = False):
    
    ## image is an ImageFile
    hr_image = Image.open(image_path).convert("RGBA")
    ## We are just arbitrarily picking the image height to be "hr_size"
    hr_width, hr_size = hr_image.size
    lr_size = int((1 - blur_strength) * (hr_size - start_res) + start_res)

    transform = transforms.Compose([transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float)])
    image_tensor = transform(hr_image)

    down = transforms.Resize(lr_size)
    up = transforms.Resize(hr_size)
    lr_image = up(down(hr_image))

    predictions = model([image_tensor[:3]])

    boxes = predictions[0]['boxes']
    for i, x in enumerate(boxes):
        x1 = int(x[0].item())
        y1 = int(x[1].item())
        x2 = int(x[2].item())
        y2 = int(x[3].item())
        ## If we're doing object blurring
        if background_blurring==False:
            #hr_image[:, y1:y2, x1:x2] = lr_image[:, y1:y2, x1:x2]
            cropped_lr_region = lr_image.crop((x1, y1, x2, y2))
            hr_image.paste(cropped_lr_region, (x1, y1, x2, y2))
        ## If we're blurring the background
        else:
            #lr_image[:, y1:y2, x1:x2] = hr_image[:, y1:y2, x1:x2]
            cropped_hr_region = hr_image.crop((x1, y1, x2, y2))
            lr_image.paste(cropped_hr_region, (x1, y1, x2, y2))
    
    if background_blurring==False:
        return hr_image
    else:
        return lr_image

#add_object_diffusion_noise(image_path="/home/nishka/Object-Hallucination-VLM/data/coco/val2014/COCO_val2014_000000579231.jpg", background_blurring=False)