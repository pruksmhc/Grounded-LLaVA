# Grounded LLaVa

SAM = Segment all objects in an image
Multi-model GPT-4/LLava/MiniGPT = Describe (describe semantics) in an image

Idea: Generate LLaVA-esque training data at scale from object detection datasets (Grounded LLaVa!)

# Requirements

Follow LlaVA installation instructions [here](https://github.com/haotian-liu/LLaVA/tree/main).

Install Segment Anything following instructions [here](https://github.com/facebookresearch/segment-anything).

# Generate Data and Train

You will need [COCO downloaded](https://cocodataset.org/#download).

```
python coco_inference/run_inference.py
```

Adjust paths in ```./scripts/finetune_lora.sh```.

```
bash /LLaVA/scripts/finetune_lora.sh
```
