from pathlib import Path
from torch.utils.data import Dataset
import torch
import numpy as np

from metro.datasets.human_mesh_tsv import MeshTSVYamlDataset


def get_paths(root):
    root = Path(root)
    return {
        'cityscapes': {
            'train_img': root / 'cityscapes/leftImg8bit/train',
            'train_anno': root / 'cityscapes/annotations/cityscapes_train.json',
            'val_img': root / 'cityscapes/leftImg8bit/val',
            'val_anno': root / 'cityscapes/annotations/cityscapes_val.json',
        },
        'cityscapes_caronly': {
            'train_img': root / 'cityscapes/leftImg8bit/train',
            'train_anno': root / 'cityscapes/annotations/cityscapes_caronly_train.json',
            'val_img': root / 'cityscapes/leftImg8bit/val',
            'val_anno': root / 'cityscapes/annotations/cityscapes_caronly_val.json',
        },
        'foggy_cityscapes': {
            'train_img': root / 'cityscapes/leftImg8bit_foggy/train',
            'train_anno': root / 'cityscapes/annotations/foggy_cityscapes_train.json',
            'val_img': root / 'cityscapes/leftImg8bit_foggy/val',
            'val_anno': root / 'cityscapes/annotations/foggy_cityscapes_val.json',
        },
        'sim10k': {
            'train_img': root / 'sim10k/VOC2012/JPEGImages',
            'train_anno': root / 'sim10k/annotations/sim10k_caronly.json',
        },
    }


class DADataset(Dataset):
    def __init__(self, yaml_file_source, yaml_file_target, is_train, scale_factor):
        self.source = MeshTSVYamlDataset(
            yaml_file_source, 
            is_train, 
            False, 
            scale_factor)
        self.target = MeshTSVYamlDataset(
            yaml_file_target,
            is_train, 
            False, 
            scale_factor)
        

    def __len__(self):
        # return max(len(self.source), len(self.target))
        return len(self.source)+len(self.target)

    def __getitem__(self, idx): #img_key, transfromed_img, meta_data
        source_key, source_img, source_meta = self.source[idx % len(self.source)]
        target_key, target_img, _ = self.target[idx % len(self.target)]
        return source_key, target_key, source_img, target_img, source_meta


def collate_fn(batch):
    source_keys, target_keys, source_imgs, target_imgs, source_metas = list(zip(*batch)) #source_targets: label of source domain

    keys = source_keys + target_keys
    imgs = source_imgs + target_imgs
    imgs = torch.stack(list(imgs), dim = 0)
    
    #to match the annotation format
    new_source_metas = {}
    for key in source_metas[0]:
        
        tmp_list=[]
        #tmp = torch.tensor([[]])  #[ [],a,b]
        for d in source_metas: 
            
            if torch.is_tensor(d[key]):
                tmp_list.append(d[key].tolist())
                
            elif isinstance(d[key], np.ndarray):
                tmp_list.append(d[key].tolist())
            else:
                tmp_list.append(d[key])
            
        if torch.is_tensor(source_metas[0][key]):          
            new_source_metas[key] = torch.tensor(tmp_list)
        elif isinstance(d[key], str):
            new_source_metas[key] = tmp_list
        else:  
            new_source_metas[key] = torch.tensor(tmp_list)

    return keys, imgs, new_source_metas


'''
def build(image_set, cfg): #image_set是'train' 或 'val', cfg
    paths = get_paths(cfg.DATASET.COCO_PATH) # cfg.DATASET.COCO_PATH是'../datasets'
    
    'cityscapes': {
            'train_img': root / 'cityscapes/leftImg8bit/train',
            'train_anno': root / 'cityscapes/annotations/cityscapes_train.json',
            'val_img': root / 'cityscapes/leftImg8bit/val',
            'val_anno': root / 'cityscapes/annotations/cityscapes_val.json',
        },
    'cityscapes_caronly': {
        'train_img': root / 'cityscapes/leftImg8bit/train',
        'train_anno': root / 'cityscapes/annotations/cityscapes_caronly_train.json',
        'val_img': root / 'cityscapes/leftImg8bit/val',
        'val_anno': root / 'cityscapes/annotations/cityscapes_caronly_val.json',
    },
    'foggy_cityscapes': {
        'train_img': root / 'cityscapes/leftImg8bit_foggy/train',
        'train_anno': root / 'cityscapes/annotations/foggy_cityscapes_train.json',
        'val_img': root / 'cityscapes/leftImg8bit_foggy/val',
        'val_anno': root / 'cityscapes/annotations/foggy_cityscapes_val.json',
    },
    'sim10k': {
        'train_img': root / 'sim10k/VOC2012/JPEGImages',
        'train_anno': root / 'sim10k/annotations/sim10k_caronly.json',
    }
    
    source_domain, target_domain = cfg.DATASET.DATASET_FILE.split('_to_') 
    # 'cityscapes_to_foggy_cityscapes' -> source_domain是'cityscapes', target_domain是'foggy_cityscapes'
    if image_set == 'val':
        return CocoDetection(
            img_folder=paths[target_domain]['val_img'], #img_folder是datasets / 'cityscapes/leftImg8bit_foggy/val'
            ann_file=paths[target_domain]['val_anno'], #ann_file是datasets / 'cityscapes/leftImg8bit_foggy/val'
            transforms=make_coco_transforms(image_set),
            return_masks=cfg.MODEL.MASKS,
            cache_mode=cfg.CACHE_MODE,
            local_rank=get_local_rank(),
            local_size=get_local_size()
        )
    elif image_set == 'train':
        if cfg.DATASET.DA_MODE == 'source_only':
            return CocoDetection(
                img_folder=paths[source_domain]['train_img'],
                ann_file=paths[source_domain]['train_anno'],
                transforms=make_coco_transforms(image_set),
                return_masks=cfg.MODEL.MASKS,
                cache_mode=cfg.CACHE_MODE,
                local_rank=get_local_rank(),
                local_size=get_local_size()
            )
        elif cfg.DATASET.DA_MODE == 'oracle':
            return CocoDetection(
                img_folder=paths[target_domain]['train_img'],
                ann_file=paths[target_domain]['train_anno'],
                transforms=make_coco_transforms(image_set),
                return_masks=cfg.MODEL.MASKS,
                cache_mode=cfg.CACHE_MODE,
                local_rank=get_local_rank(),
                local_size=get_local_size()
            )
        elif cfg.DATASET.DA_MODE == 'uda':
            return DADataset(
                source_img_folder=paths[source_domain]['train_img'],#source_img_folder是datasets / 'cityscapes/leftImg8bit/train'
                source_ann_file=paths[source_domain]['train_anno'],#source_ann_file是datasets / 'cityscapes/annotations/cityscapes_train.json'
                target_img_folder=paths[target_domain]['train_img'],#target_img_folder是datasets / 'cityscapes/leftImg8bit_foggy/train'
                target_ann_file=paths[target_domain]['train_anno'],#target_ann_file是datasets / 'cityscapes/annotations/foggy_cityscapes_train.json'
                transforms=make_coco_transforms(image_set),
                return_masks=cfg.MODEL.MASKS,
                cache_mode=cfg.CACHE_MODE,
                local_rank=get_local_rank(),
                local_size=get_local_size()
            )
        else:
            raise ValueError(f'Unknown argument cfg.DATASET.DA_MODE {cfg.DATASET.DA_MODE}')
    raise ValueError(f'unknown image set {image_set}')
'''