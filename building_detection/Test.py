import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from model import EfficientFPN
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
import albumentations as A
from skimage.transform import resize

albu_dev = A.Compose([
    A.Normalize(),
    ToTensorV2(),
])

model = EfficientFPN(encoder_name='efficientnet-b2', use_context_block=True, use_mish=True)#, use_attention=True)
state = torch.load('models/model-b2-2.pth', map_location=lambda storage, loc: storage)

model.load_state_dict(state["state_dict"])

#device = torch.device("cuda:0")
device = torch.device("cpu")
model = model.to(device)
model = model.eval()

def run_test(fname, downscale_f = 1):
    block = 1024

    #import time
    #tic = time.time()

    im = Image.open(fname).convert('RGB')
    new_0 = int(np.round(downscale_f*np.size(im)[0]))
    new_1 = int(np.round(downscale_f*np.size(im)[1]))
    img = im.resize((new_0,new_1))
    nimg = np.array(img)
    data = albu_dev(image=nimg)
    nimg = data['image'].float()
    p_width = block - nimg.size(2)%block
    p_height = block - nimg.size(1)%block
    nimg = F.pad(nimg, (0, p_width, 0, p_height))

    patches = nimg.data.unfold(0, 3, 3).unfold(1, block, block).unfold(2, block, block)
    p = patches.reshape(-1,3,block,block)
    out_size=list(p.size())
    out_size.pop(1)
    outputs = torch.empty(out_size)
    dice = torch.empty(out_size)
    with torch.no_grad():
        for i in range(p.size(0)):
            pi = p[i,:,:,:].unsqueeze(0)
            pi= pi.to(device)
            out, d, cls, scale = model(pi)
            out_log = F.softmax(out, 1)
            out_log = F.interpolate(out_log, size=(block, block), mode='bilinear', align_corners=False)
            out_log = out_log[:, 1].detach()#.numpy()
            outputs[i,:,:]=out_log

            d = torch.sigmoid(d)
            d = F.interpolate(d, size=(block, block), mode='bilinear', align_corners=False)
            d[d < 0.5] = 0
            d[d >= 0.5] = 1
            d = d[0, 0, :, :].detach()#.numpy() * 255
            dice[i,:,:] = d
    finalsize=list(patches.size())
    finalsize[3]=1
    p_out = outputs.reshape(finalsize)
    p_d = dice.reshape(finalsize)
    out_sz=list(nimg.size())

    re_o = p_out.permute(0, 3, 1, 4, 2, 5).contiguous().reshape(out_sz[1:])#.view_as(nimg)
    re_o = re_o.numpy()

    re_d = p_d.permute(0, 3, 1, 4, 2, 5).contiguous().reshape(out_sz[1:])#.view_as(nimg)
    re_d = re_d.numpy() * 255
    re_d = re_d.astype(np.uint8)
    #toc = time.time()
    #print(toc-tic)
    plt.imshow(np.array(img))
    plt.show()
    plt.imshow(re_o[:new_1,:new_0])
    plt.show()
    plt.imshow(re_d[:new_1,:new_0])
    plt.show()

    seg = re_d[:new_1,:new_0]
    seg = resize(seg, (np.size(im)[1],np.size(im)[0]))
    seg = seg * 255
    seg = seg.astype(np.uint8)

    im = np.array(im)
    im_mask = np.zeros_like(im)
    idx=(seg==255)
    im_mask[idx]=im[idx]
    plt.imshow(im_mask)
    return seg, im_mask

fname = 'neighborhood_image.jpg'
seg, mask = run_test(fname, downscale_f = 1) # f = 0.25 for danny's images
print("Test complete")

im_mask = np.array(mask, dtype='uint8')
im_mask_ = Image.fromarray(im_mask)
im_mask_.save("./neighborhood_buildings.png")

im_seg = np.array(seg, dtype='uint8')
im_seg_ = Image.fromarray(im_seg)
im_seg_.save("./neighborhood_buildings_seg.png")
