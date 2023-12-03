import pandas as pd
import numpy as np
import cv2
from PIL import Image
import os, os.path
import torch
from carvekit.api.high import HiInterface

interface = HiInterface(object_type="object",  # Can be "object" or "hairs-like" - object better
                        batch_size_seg=5,
                        batch_size_matting=1,
                        device='cuda' if torch.cuda.is_available() else 'cpu',
                        seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net - Tracer B7 better
                        matting_mask_size=2048,
                        trimap_prob_threshold=250,
                        trimap_dilation=30,
                        trimap_erosion_iters=5,
                        fp16=False)

def remove_border(img):
    img_cv = np.array(img)
    lower = np.array([0,99,99,0])  #-- Lower range --
    upper = np.array([40,255,255,255])  #-- Upper range --
    mask = cv2.inRange(img_cv, lower, upper)
    res = cv2.bitwise_and(img_cv, img_cv, mask= mask)  #-- Contains pixels having the gray color--
    diff = cv2.subtract(img_cv,res)
    return Image.fromarray(diff)

def remove_background_df(df):
   for i in range(len(df)):
        img_wo_bg = interface([df['images'][i]])[0]
        img_wo_bg = remove_border(img_wo_bg)
        background = Image.new('RGBA', img_wo_bg.size, (0,0,0))
        img = Image.alpha_composite(background, img_wo_bg)
        print('inserted ' + df['Roll_Pic'][i])
        location = '/Users/cameronhudson/Documents/Masters/Research/Snow/new_pics/' + df['Roll_Pic'][i] + '.png'
        img.save(location)

def main():
    LH = pd.read_csv('LHDataClean.csv') 
    # LH = LH.sort_values(by=['Roll Number'])
    # LH['Pic Number'] = LH['Pic Number'].astype(str).str.zfill(2)
    # LH['Roll Number'] = LH['Roll Number'].astype(str).str.zfill(2)

    # LH['Roll_Pic'] = LH['Roll Number'].astype(str) + "_" + LH['Pic Number'].astype(str)
    LH = LH.reset_index(drop=True)

    #LH = LH.drop(['Roll Number', 'Pic Number'], axis=1)

    imgs = {}
    path = "/Users/cameronhudson/Documents/Masters/Research/Snow/LH_pics"
    valid_images = [".jpg",".gif",".png",".tga"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        imgs[f[15:20]] = Image.open(os.path.join(path,f))

    img_df = pd.DataFrame(imgs.items(), columns=['Roll_Pic', 'images'])
    LH = pd.concat([LH.set_index('Roll_Pic'),img_df.set_index('Roll_Pic')], axis=1, join='inner').reset_index()

    remove_background_df(LH)

if __name__ == "__name__":
    main()
