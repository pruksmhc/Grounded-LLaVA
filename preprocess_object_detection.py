import json
from tqdm import tqdm
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

def collate_jsons():
    import os
    import json

    # Base directory path
    base_dir = "/scratch/shared/beegfs/sagar/slurm_outputs/"

    # Dictionary to store the collected data
    collected_data = {}

    # Loop through each directory by varying the IDX value
    for idx in tqdm(range(20)):  # As IDX goes from 0 to 19 inclusive
        current_dir = os.path.join(base_dir, f"10810_{idx}")
        
        # Check if directory exists
        if os.path.exists(current_dir):
            # List all files in the directory
            for filename in os.listdir(current_dir)[:10]:
                # Check if the file is a JSON file
                if filename.endswith(".json"):
                    # Construct the complete file path
                    filepath = os.path.join(current_dir, filename)
                    
                    # Read the content of the JSON file
                    with open(filepath, 'r') as json_file:
                        content = json.load(json_file)
                    
                    # Extract the image ID from the filename by removing the extension
                    image_id = os.path.splitext(filename)[0]
                    
                    # Store the content in the collected_data dictionary
                    collected_data[image_id] = content

    # Write the collected data into a single JSON file
    output_path = "annots.json"
    with open(output_path, 'w') as output_file:
        json.dump(collected_data, output_file, indent=4)
    
    return collected_data


sam_outputs = collate_jsons()
final_res = []
for image_id, values in sam_outputs.items():
    final_res.extend(preprocess_obj_detection(image_id, values))
json.dump(final_res, open("llava_object_detection.json", "w"), indent=4)
