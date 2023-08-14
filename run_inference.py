from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import cv2
import json
import os
from tqdm import tqdm

import submitit
import argparse
import uuid
from pathlib import Path
import numpy as np

def parse_args():

    parser = argparse.ArgumentParser('COCOSam', add_help=False)
    parser.add_argument("--ngpus", default=1, type=int, help="Number of gpus to request on each node")
    parser.add_argument("--nodes", default=1, type=int, help="Number of nodes to request")
    parser.add_argument("--njobs", default=1, type=int, help="Number of nodes to request")
    parser.add_argument("--timeout", default=4320, type=int, help="Duration of the job")
    parser.add_argument("--output_dir", default='/scratch/shared/beegfs/sagar/slurm_outputs', type=str, help="Where stdout and stderr will write to")

    parser.add_argument("--partition", default="slurm", type=str, help="Partition where to submit")
    parser.add_argument('--comment', default="", type=str,
                        help='Comment to pass to scheduler, e.g. priority message')
    return parser.parse_args()

def get_shared_folder() -> Path:
    if Path('/scratch/shared/beegfs/sagar/slurm_outputs').is_dir():
        p = Path('/scratch/shared/beegfs/sagar/slurm_outputs')
        return p
    raise RuntimeError("No shared folder available")

def get_job_dir(root, job_id=None):
    job_folder = "%j" if job_id is None else str(job_id)
    return os.path.join(root, "jobs", job_folder)

def get_init_file():
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file

class COCOSAMInference:

    def __init__(
            self,
            coco_image_root,
            coco_annot_path,
            output_dir,
            model_type = "vit_h", 
            sam_checkpoint = "/work/sagar/pretrained_models/sam/sam_vit_h_4b8939.pth",
            device = "cuda"
            ):

        self.device = device
        self.sam_checkpoint = sam_checkpoint
        self.output_dir = output_dir
        self.model_type = model_type
        self.coco_image_root = coco_image_root

        with open(coco_annot_path, 'r') as f:
            self.coco_annots = json.load(f)

        self.coco_category_ids_to_str = {
            x['id']: x['name'] for x in self.coco_annots['categories']
        }

    def __call__(self, idxs_to_process):
        
        job_env = submitit.JobEnvironment()
        output_dir = os.path.join(str(self.output_dir).strip('%j'), job_env.job_id)

        assert os.path.exists(output_dir)

        print('Loading SAM model...')
        sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)

        sam.to(device=self.device)
        self.sam = sam
        self.mask_generator = SamAutomaticMaskGenerator(sam, output_mode='coco_rle')

        print('Starting inference...')
        for idx in tqdm(idxs_to_process):
            output = self._do_inference(idx)

            sample_output_path = os.path.join(output_dir, output['image_metadata']['file_name'].replace('.jpg', '.json'))
            with open(sample_output_path, 'w') as f:
                json.dump(output, f, indent=4)
    
    def _do_inference(self, idx):
    
        # Get COCO image
        img_meta = self.coco_annots['images'][idx]
        file_name = img_meta['file_name']
        img_path = os.path.join(self.coco_image_root, file_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get SAM outputs
        sam_outputs = self.mask_generator.generate(image)

        # Get original annotations
        coco_annots = self.coco_annots['annotations'][idx]

        for seg_annot in coco_annots['segments_info']:
            seg_annot['category_name'] = self.coco_category_ids_to_str[seg_annot['category_id']]

        output = {
            'image_metadata': img_meta,
            'coco_annotations': coco_annots,
            'sam_outputs': sam_outputs
        }

        return output

if __name__ == '__main__':

    args = parse_args()
    if args.output_dir == "":
        args.output_dir = get_shared_folder() / "%j"
    else:
        args.output_dir = Path(args.output_dir) / "%j"

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    executor = submitit.AutoExecutor(folder=args.output_dir, slurm_max_num_timeout=30)

    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.timeout

    partition = args.partition
    kwargs = {}
    if args.comment:
        kwargs['slurm_comment'] = args.comment

    executor.update_parameters(
        mem_gb=75 * num_gpus_per_node,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,  # one task per GPU
        cpus_per_task=10,
        exclude='gnodec1',
        nodes=nodes,
        timeout_min=timeout_min,  # max is 60 * 72
        # Below are cluster dependent parameters
        slurm_partition=partition,
        slurm_signal_delay_s=120,
        **kwargs
    )

    executor.update_parameters(name="sam_inf")

    output_path = '/users/sagar/misc/segment-anything/coco_inference/test_2.json'
    generator = COCOSAMInference(
        coco_image_root='/scratch/shared/beegfs/shared-datasets/COCO/COCO2017/train2017',
        coco_annot_path='/work/sagar/datasets/COCO_panoptic/annotations/panoptic_train2017.json',
        output_dir=args.output_dir
    )
    idxs_to_process = np.arange(len(generator.coco_annots['annotations']))

    if args.partition == 'slurm':

        idxs_to_process_chunk = np.array_split(idxs_to_process, args.njobs)

        # Launch jobs
        jobs = []
        with executor.batch():
            for idxs in tqdm(idxs_to_process_chunk):
                job = executor.submit(generator, idxs_to_process=idxs)
                jobs.append(job)
        
        print(f"Submitted job_id: {jobs[0].job_id}")
    
    else:

        generator(idxs_to_process=idxs_to_process)
    
