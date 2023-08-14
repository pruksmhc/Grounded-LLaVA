import json
# TODO: multiprocessing 
USER_TEAMPLTE = "Where is the {object_name}? They might be at these locations {sam_bbox_outputs}"
ASSISTANT_TEMPLATE = "The {object_name} is at {bbox_output}"

def preprocess_obj_detection(image_id, image_example):
    """
    Output:
    "image": "000000337638.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "where is the people? They might be at these locations: [0.681, 0.242, 0.774, 0.694], []"
      },
      {
        "from": "assistant",
        "value": "The X is at [0.681, 0.242, 0.774, 0.694]"
      },
    ]}
    """
    final_obj = []
    obj_to_bbox = {}
    for annotation in image_example["coco_annotations"]["segments_info"]:
        if annotation["category_name"] not in obj_to_bbox:
            obj_to_bbox[annotation["category_name"]] = []
        obj_to_bbox[annotation["category_name"]].append(str(annotation["bbox"]))
    sam_bbox_outputs = ",".join([str(x["bbox"]) for x in image_example["sam_outputs"]])
    for obj_name, gold_bboxes in obj_to_bbox.items():
        conversation = {"image": image_id, "conversations": []}
        # TODO: pluralize based on count
        gold_bboxes[-1] = f" and {gold_bboxes[-1]}"
        gold_bboxes = ",".join(gold_bboxes)
        conversation['conversations'].append({"from": "human", "value":USER_TEAMPLTE.format(object_name=obj_name, sam_bbox_outputs=sam_bbox_outputs)})
        conversation['conversations'].append({"from": "gpt4", "value":  ASSISTANT_TEMPLATE.format(object_name=obj_name, bbox_output=gold_bboxes)})
        final_obj.append(conversation)
    return final_obj

sam_outputs = json.load(open("sam_annots_test_final.json", "r"))
final_res = []
for image_id, values in sam_outputs.items():
    final_res.extend(preprocess_obj_detection(image_id, values))
json.dump(final_res, open("llava_object_detection.json", "w"))
