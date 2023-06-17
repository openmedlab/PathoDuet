import csv
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from sklearn.utils import shuffle
from multiprocessing import Pool

from openslide import OpenSlide

def get_tissue_mask(slide_path):
    '''
    slide_path: path for each slide
    '''
    
    slide = AllSlide(slide_path)
    thumb = slide.read_region((0, 0), slide.level_count-2, slide.level_dimensions[-2])
    
    img_RGB = np.array(thumb)
    slide_lv = cv2.cvtColor(img_RGB, cv2.COLOR_RGBA2RGB)
    slide_lv = cv2.cvtColor(slide_lv, cv2.COLOR_BGR2HSV)
    slide_lv = slide_lv[:, :, 1]

    _, tissue_mask = cv2.threshold(slide_lv, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    tissue_mask[tissue_mask != 0] = 1
    
    _, label = cv2.connectedComponents(tissue_mask)
    tissue_mask = remove_small_objects(label, min_size=64)
    tissue_mask = np.array(tissue_mask, np.uint8)
    
    return Image.fromarray(tissue_mask)


def get_mil_data(slide_path, slide_ind, savepath, dataset, num_region=200, ps_region=1024, region_level=1,
                    num_patch=9, ps_patch=256, threshold_region=220, threshold_patch=240):
    '''
    slide_path: path for each slide
    slide_id: index of the slide
    savepath: path to outpath/tcga-xxxx/
    dataset: index of tcga datasets
    num_region: number of regions in a wsi
    ps_region: region size
    region_level: at which level the region is sampled (patch is sampled at level-1)
    num_patch: maximum number of patches in a region
    ps_patch: patch size
    threshold_*: used to remove the patch from background
    '''
    
    # generate tissue mask
    slide = OpenSlide(slide_path)
    # slide_name = slide_path.split('/')[-1].split('.')[0]
    out_file_region = os.path.join(savepath, 'region')
    out_file_patch = os.path.join(savepath, 'patch')

    tissue_mask = get_tissue_mask(slide_path)
    if tissue_mask == None:
        print(f'{slide_ind} Skipped......')
        return

    w, h = slide.level_dimensions[region_level]
    downsample_region = int(slide.level_downsamples[region_level])
    downsample_patch = int(slide.level_downsamples[region_level-1])
    rs_w = int(w/ps_region); rs_h = int(h/ps_region)
    delta_hw = (128, ps_region*downsample_region-ps_patch*downsample_patch-128)

    tissue_mask = np.array(tissue_mask.resize((rs_w, rs_h)))
    h_list, w_list = np.where(tissue_mask != 0)

    idx = shuffle(list(range(len(h_list))))
    h_select = h_list[idx] * ps_region * downsample_region
    w_select = w_list[idx] * ps_region * downsample_region

    # random select regions
    region_count = 0
    for i in range(len(h_select)):
        region = slide.read_region((w_select[i], h_select[i]), region_level, (ps_region, ps_region))

        if np.mean(region) < threshold_region:
            region.resize((ps_patch, ps_patch)).save(os.path.join(out_file_region, f'{slide_ind}_{i}.png'))
            region_count += 1

            # random select patchs
            for j in range(num_patch):
                delta_h = np.random.randint(delta_hw[0], delta_hw[1])
                delta_w = np.random.randint(delta_hw[0], delta_hw[1])
                patch = slide.read_region((w_select[i]+delta_w, h_select[i]+delta_h), region_level-1, (ps_patch, ps_patch))
                if np.mean(patch) < threshold_patch:
                    patch.save(os.path.join(out_file_patch, f'{slide_ind}_{i}_{j}.png'))

        if region_count >= num_region:
            break

    print(f'{slide_ind} Done......')

slide_list = glob.glob('./*.svs')

with Pool(processes=8) as p:
    p.map(get_mil_data, slide_list)