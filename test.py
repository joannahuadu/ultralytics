from ultralytics import YOLO
import json
import os

# Load a pretrained YOLO11n model
model = YOLO("/mnt/data1/workspace/wmq/ultralytics/runs/detect/train10/weights/best.pt")
jsonfile = "/mnt/data1/workspace/data/data/qiyuan2/instances_test.json"
outfile_prefix = "/mnt/data1/workspace/wmq/ultralytics/runs/detect/train10/epoch100/conf0.0001/pred_0619_cucu"
with open(jsonfile, 'r') as f:
    json_data = json.load(f)

images = json_data['images']
categories = json_data['categories']
cat_ids = {4: 1, 2: 2, 6: 3, 5: 4, 8: 5, 9: 6, 1: 7, 3: 8, 0: 9, 7: 10}
results = model.predict("/mnt/data1/workspace/data/data/qiyuan2/test/images", save=False, imgsz=1024, conf=0.0001)
bbox_json_results = []
instance_id = 0
for res in results:
    res.boxes.xywh[:, 0] =  res.boxes.xywh[:, 0] - res.boxes.xywh[:, 2]/2
    res.boxes.xywh[:, 1] =  res.boxes.xywh[:, 1] - res.boxes.xywh[:, 3]/2
    file_name = os.path.basename(res.path)  # 提取文件名 'test_000000.jpg'
    image_id = int(file_name.replace('.jpg', '').split('_')[-1]) + 1
    labels = res.boxes.cls
    bboxes = res.boxes.xywh
    scores = res.boxes.conf
    for cls, score, box in zip(labels, scores, bboxes):
        instance_id+=1
        data = dict()
        data['image_id'] = image_id
        data['id'] = instance_id
        data['bbox'] = box.tolist()
        data['score'] = float(score)
        data['category_id'] = cat_ids[int(cls)]
        bbox_json_results.append(data)

coco_format_results = {
    "images": images,
    "annotations": bbox_json_results,
    "categories": categories,
}
result_files = f'{outfile_prefix}/pred.json'

os.makedirs(os.path.dirname(result_files), exist_ok=True)

with open(result_files, 'w') as f:
    json.dump(coco_format_results, f)