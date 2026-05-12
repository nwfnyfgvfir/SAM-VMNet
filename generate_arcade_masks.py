import os
import cv2
import numpy as np
from pycocotools.coco import COCO

def generate_correct_masks():
    # ================= 配置路径 =================
    json_file = './data/vessel/val/annotations/val.json'   
    mask_output_dir = './data/vessel/val/masks'
    os.makedirs(mask_output_dir, exist_ok=True)
    # ============================================
    
    print(f"正在读取标注文件: {json_file} ...")
    coco = COCO(json_file)
    img_ids = coco.getImgIds()
    
    # 定义形态学“膨胀”的核（3x3）
    # 它的作用非常温和：只让白色的血管向外胖 1 个像素，刚好能连上微小的断点
    kernel = np.ones((3, 3), np.uint8)

    print(f"共发现 {len(img_ids)} 张有标注的图片。开始生成精准掩码...")

    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        width = img_info['width']
        height = img_info['height']
        
        mask_canvas = np.zeros((height, width), dtype=np.uint8)
        
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        for ann in anns:
            # 【核心修复】绝对不自己画！只用官方 API 渲染真实的像素级掩码
            pixel_mask = coco.annToMask(ann)
            mask_canvas = np.maximum(mask_canvas, pixel_mask)
                
        # 将 0和1 转换为 0和255(黑白)
        mask_canvas = mask_canvas * 255
        
        # 【修复断连】进行一次轻微的膨胀操作，把相邻的断点“挤”在一起
        mask_canvas = cv2.dilate(mask_canvas, kernel, iterations=1)
        
        save_path = os.path.join(mask_output_dir, file_name)
        cv2.imwrite(save_path, mask_canvas)
        
    print("✅ 纯净版掩码生成完毕！它们已保存在:", mask_output_dir)

if __name__ == '__main__':
    generate_correct_masks()