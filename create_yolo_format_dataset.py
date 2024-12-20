import os
import cv2
import glob
import json
import shutil
from pprint import pprint

if __name__ == '__main__':
    root_path = "data/football_test"
    output_path = "football_yolo_dataset"
    is_train = False

    video_paths = list(glob.iglob("{}/*/*.mp4".format(root_path)))
    anno_paths = list(glob.iglob("{}/*/*.json".format(root_path)))
    video_wo_ext = [video_path.replace(".mp4", "") for video_path in video_paths]
    ann_wo_ext = [anno_path.replace(".json", "") for anno_path in anno_paths]
    paths = list(set(video_wo_ext) & set(ann_wo_ext))
    mode = "train" if is_train else "val"

    if not os.path.isdir(output_path) and is_train:
        os.makedirs(output_path)
        os.makedirs(os.path.join(output_path, "images"))
        os.makedirs(os.path.join(output_path, "images", mode))
        os.makedirs(os.path.join(output_path, "labels"))
        os.makedirs(os.path.join(output_path, "labels", mode))
    elif not is_train:
        os.makedirs(os.path.join(output_path, "images", mode))
        os.makedirs(os.path.join(output_path, "labels", mode))

    for idx, path in enumerate(paths):
        video = cv2.VideoCapture("{}.mp4".format(path))
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        with open("{}.json".format(path), "r") as json_file:
            json_data = json.load(json_file)
        # if num_frames != len(json_data["images"]):
        #     print("Something went wrong with {}".format(path))
        #     paths.remove(path)

        width = json_data["images"][0]["width"]
        height = json_data["images"][0]["height"]
        all_objects = [{"image_id": obj["image_id"], "bbox": obj["bbox"], "category_id": obj["category_id"]} for obj in
                       json_data["annotations"] if obj["category_id"] in [3, 4]]
        all_balls = [obj for obj in json_data["annotations"] if obj["category_id"] == 3]
        frame_counter = 0
        while video.isOpened():
            print(idx, frame_counter)
            flag, frame = video.read()
            if not flag:
                break
            cv2.imwrite(os.path.join(output_path, "images", mode, "{}_{}.jpg".format(idx, frame_counter)), frame)
            current_object = [obj for obj in all_objects if obj["image_id"] - 1 == frame_counter]
            with open(os.path.join(output_path, "labels", mode, "{}_{}.txt".format(idx, frame_counter)), "w") as  f:
                for obj in current_object:
                    xmin, ymin, w, h = obj["bbox"]
                    xmin /= width
                    w /= width
                    ymin /= height
                    h /=height
                    if obj["category_id"] == 4:
                        category = 0
                    else:
                        category = 1
                    f.write("{} {:06f} {:06f} {:06f} {:06f}\n".format(category, xmin+w/2, ymin+h/2, w, h))
            frame_counter += 1
