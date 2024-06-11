# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import copy
from ikomia import core, dataprocess, utils
from infer_mobile_segment_anything.MobileSAM.mobile_encoder.setup_mobile_sam import setup_model
from segment_anything import SamPredictor, SamAutomaticMaskGenerator
from infer_mobile_segment_anything.draw_graphics import DrawingGraphics
import numpy as np
import torch
import cv2
import os
import json
import requests
from PyQt5.QtWidgets import QApplication


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferMobileSegmentAnythingParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.points_per_side = 32
        self.points_per_batch = 64
        self.stability_score_thres = 0.95
        self.box_nms_thres = 0.7
        self.iou_thres = 0.88
        self.crop_n_layers = 0
        self.crop_nms_thres = 0.70
        self.crop_overlap_ratio = float(512 / 1500)
        self.crop_n_points_downscale_factor = 1
        self.min_mask_region_area = 0
        self.input_size_percent = 100
        self.mask_id = 1
        self.draw_graphic_input = False
        self.input_point = ''
        self.input_box = ''
        self.input_point_label = ''
        self.cuda = torch.cuda.is_available()
        self.update = False

    def set_values(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.points_per_side = int(param_map["points_per_side"])
        self.points_per_batch = int(param_map["points_per_batch"])
        self.iou_thres = float(param_map["iou_thres"])
        self.stability_score_thres = float(param_map["stability_score_thres"])
        self.box_nms_thres = float(param_map["box_nms_thres"])
        self.crop_n_layers = int(param_map["crop_n_layers"])
        self.crop_nms_thres = float(param_map["crop_nms_thres"])
        self.crop_overlap_ratio = float(param_map["crop_overlap_ratio"])
        self.crop_n_points_downscale_factor = int(param_map["crop_n_points_downscale_factor"])
        self.min_mask_region_area = int(param_map["min_mask_region_area"])
        self.input_size_percent = int(param_map["input_size_percent"])
        self.mask_id = int(param_map["mask_id"])
        self.draw_graphic_input = utils.strtobool(param_map["draw_graphic_input"])
        self.input_point = param_map['input_point']
        self.input_point_label = param_map['input_point_label']
        self.input_box = param_map['input_box']
        self.cuda = utils.strtobool(param_map["cuda"])
        self.update = True

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = {
            "points_per_side": str(self.points_per_side),
            "points_per_batch": str(self.points_per_batch),
            "iou_thres": str(self.iou_thres),
            "stability_score_thres": str(self.stability_score_thres),
            "box_nms_thres": str(self.box_nms_thres),
            "crop_n_layers": str(self.crop_n_layers),
            "crop_overlap_ratio": str(self.crop_overlap_ratio),
            "crop_nms_thres": str(self.crop_nms_thres),
            "crop_n_points_downscale_factor": str(self.crop_n_points_downscale_factor),
            "min_mask_region_area": str(self.min_mask_region_area),
            "input_size_percent": str(self.input_size_percent),
            "mask_id": str(self.mask_id),
            "draw_graphic_input": str(self.draw_graphic_input),
            "input_point": str(self.input_point),
            "input_point_label": str(self.input_point_label),
            "input_box": str(self.input_box),
            "cuda": str(self.cuda)
        }
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferMobileSegmentAnything(dataprocess.CSemanticSegmentationTask):

    def __init__(self, name, param):
        dataprocess.CSemanticSegmentationTask.__init__(self, name)

        # Create parameters class
        if param is None:
            self.set_param_object(InferMobileSegmentAnythingParam())
        else:
            self.set_param_object(copy.deepcopy(param))

        self.mobile_sam = None
        self.input_point = None
        # foreground point
        self.input_label = np.array([1])
        self.input_box = None
        self.multi_mask_out = True
        self.device = torch.device("cpu")
        self.model_sam_url = 'https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/weights/mobile_sam.pt'
        self.model_weight_folder = os.path.join(
                                    os.path.dirname(__file__),
                                    'MobileSAM',
                                    'weights')
        self.checkpoint_path = os.path.join(self.model_weight_folder, 'mobile_sam.pt')

    def get_progress_steps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def download_model(self):
        response = requests.get(self.model_sam_url)
        if not os.path.isdir(self.model_weight_folder):
            os.mkdir(self.model_weight_folder)
        with open(self.checkpoint_path, "wb") as f:
            f.write(response.content)

    def infer_mask_generator(self, image, mobilesam):
        # Get parameters :
        param = self.get_param_object()

        mask_generator = SamAutomaticMaskGenerator(
            model=mobilesam,
            points_per_side=param.points_per_side,  # number of points to be sampled along one side of the image
            points_per_batch=param.points_per_batch,  # number of points to be sampled in one batch
            pred_iou_thresh=param.iou_thres,  # predicted mask quality
            stability_score_thresh=param.stability_score_thres,  # cutoff used to binarize the mask predictions
            box_nms_thresh=param.box_nms_thres,  # box IoU cutoff (filter duplicate masks)
            crop_n_layers=param.crop_n_layers,  # mask prediction will be run again on crops of the image
            crop_overlap_ratio=param.crop_overlap_ratio,
            crop_nms_thresh=param.crop_nms_thres,
            crop_n_points_downscale_factor=param.crop_n_points_downscale_factor,
            min_mask_region_area=param.min_mask_region_area  # post-process remove disconnected regions
        )

        # Generate mask
        mask_output = None
        results = mask_generator.generate(image)

        if len(results) > 0:
            mask_output = np.zeros((
                        results[0]["segmentation"].shape[0],
                        results[0]["segmentation"].shape[1]
                        ))
            for i, mask_bool in enumerate(results):
                i += 1
                mask_output = mask_output + mask_bool["segmentation"] * i

        else:
            print("No mask predicted, increasing the number of points per side may help")

        return mask_output

    def infer_predictor(self, graph_input, src_image, resizing, pred, box_list, point):
        # Get parameters :
        param = self.get_param_object()

        # Get input from graphics (API)
        if param.draw_graphic_input:
            app = QApplication([])
            drawing_app = DrawingGraphics(src_image)
            drawing_app.show()
            app.exec_()
            app.quit()
            if len(drawing_app.boxes) > 0:
                self.input_box = np.array(drawing_app.boxes)
                self.input_label = np.array([0])  # background point
                self.multi_mask_out = False

            if len(drawing_app.point) > 0:
                self.input_point = np.array([drawing_app.point])

        # Get input from coordinate prompt in STUDIO
        elif box_list or point:
            if box_list:
                box_list = json.loads(box_list)
                self.input_box = np.array(box_list)
                self.input_box = self.input_box * resizing
                self.input_label = np.array([0])  # background point
                self.multi_mask_out = False
 
            if point:
                point = json.loads(point)
                self.input_point = np.array([point])
                self.input_point = self.input_point * resizing

        # Get input from drawn graphics in STUDIO
        else:
            graphics = graph_input.get_items()  # Get list of input graphics items.
            box = []
            point = []
            for i, graphic in enumerate(graphics):
                bboxes = graphics[i].get_bounding_rect()  # Get graphic coordinates
                if graphic.get_type() == core.GraphicsItem.RECTANGLE:  # rectangle
                    x1 = bboxes[0]*resizing
                    y1 = bboxes[1]*resizing
                    x2 = (bboxes[2]+bboxes[0])*resizing
                    y2 = (bboxes[3]+bboxes[1])*resizing
                    box.append([x1, y1, x2, y2])
                    self.input_box = np.array(box)
                    self.input_label = np.array([0])  # background point
                    self.multi_mask_out = False
                if graphic.get_type() == core.GraphicsItem.POINT:  # point
                    x1 = bboxes[0]*resizing
                    y1 = bboxes[1]*resizing
                    point.append([x1, y1])
                    self.input_point = np.array(point)

        pred = SamPredictor(pred)
        # Calculate the necessary image embedding
        pred.set_image(src_image)
        mask_output = np.zeros(src_image.shape[:2])

        if self.input_box is None and self.input_point is None:
            print('Use graphic inputs or set the parameters draw_graphic_input to False')
        elif self.input_box is not None and len(self.input_box) > 1:
            # Inference from multiple boxes
            self.multi_mask_out = False
            input_boxes = torch.tensor(self.input_box, device=self.device)
            transformed_boxes = pred.transform.apply_boxes_torch(
                                                input_boxes,
                                                src_image.shape[:2]
                                                )
            masks, _, _ = pred.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
                )

            mask_output = np.zeros((
                            src_image.shape[0],
                            src_image.shape[1]
                            ))

            for i, mask_bool in enumerate(masks):
                mask = mask_bool.cpu().numpy()[0]
                i += 1
                mask_output = mask_output + mask * i
        elif self.input_point is not None and self.input_box is None:
            # Inference from points
            if len(self.input_point) == 1:
                masks, _, _ = pred.predict(
                    point_coords=self.input_point,
                    point_labels=self.input_label,
                    multimask_output=True,
                )
                mask_output = masks[param.mask_id-1]
            
            if len(self.input_point) > 1:
                if param.input_point_label: 
                    self.input_label = json.loads(param.input_point_label)
                    self.input_label = np.array(self.input_label)
                    if len(self.input_label) != self.input_label:  # Edit input label if the user makes a mistake
                        self.input_label = np.ones(len(self.input_point))
                else:
                    self.input_label = np.ones(len(self.input_point))  # Automatically generate input labels

                masks, _, _ = pred.predict(
                    point_coords=self.input_point,
                    point_labels=self.input_label,
                    multimask_output=True,
                )
                mask_output = masks[param.mask_id-1]
        elif self.input_point is None and len(self.input_box) == 1:
            # Inference from a single box
            masks, _, _ = pred.predict(
                point_coords=None,
                point_labels=None,
                box=self.input_box[None, :],
                multimask_output=False,
            )
            mask_output = masks[0]
        elif self.input_point is not None and len(self.input_box) == 1:
            # Inference from a single box and a single point
            masks, _, _ = pred.predict(
                point_coords=self.input_point,
                point_labels=np.array([0]),
                box=self.input_box,
                multimask_output=False,
            )
            mask_output = masks[0]
        else:
            print("Please select a point and/or a box")

        return mask_output

    def run(self):
        # Core function of your process
        # Call begin_task_run() for initialization
        self.begin_task_run()
        print("Running...")
        # Get input
        task_input = self.get_input(0)
        # Get parameters :
        param = self.get_param_object()
        # Get image from input/output (numpy array):
        src_image = task_input.get_image()

        # Resize image
        ratio = param.input_size_percent / 100
        h_orig, w_orig = src_image.shape[0], src_image.shape[1]

        if param.input_size_percent < 100:
            width = int(src_image.shape[1] * ratio)
            height = int(src_image.shape[0] * ratio)
            dim = (width, height)
            src_image = cv2.resize(src_image, dim, interpolation=cv2.INTER_LINEAR)

        # Load model
        if param.update or self.mobile_sam is None:
            self.device = torch.device("cuda") if param.cuda and torch.cuda.is_available() else torch.device("cpu")
            if not os.path.isfile(self.checkpoint_path):
                self.download_model()
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.mobile_sam = setup_model()
            self.mobile_sam.load_state_dict(checkpoint, strict=True)
            self.mobile_sam.to(device=self.device)
            self.mobile_sam.eval()
            param.update = False

        graph_input = self.get_input(1)

        if graph_input.is_data_available() or param.draw_graphic_input or param.input_box or param.input_point:
            mask = self.infer_predictor(
                graph_input=graph_input,
                src_image=src_image,
                resizing=ratio,
                pred=self.mobile_sam,
                box_list=param.input_box,
                point=param.input_point
            )
        else:
            mask = self.infer_mask_generator(src_image, self.mobile_sam)

        if mask is None:
            raise(RuntimeError, "Failed to generate segmentation mask.")

        mask = mask.astype("uint8")
        if param.input_size_percent < 100:
            mask = cv2.resize(
                            mask,
                            (w_orig, h_orig),
                            interpolation=cv2.INTER_NEAREST
            )

        self.get_output(0)
        self.set_mask(mask)

        # Step progress bar (Ikomia Studio):
        self.emit_step_progress()

        # Call end_task_run() to finalize process
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferMobileSegmentAnythingFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_mobile_segment_anything"
        self.info.short_description = "Inference for Mobile Segment Anything Model (SAM)."
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Segmentation"
        self.info.version = "1.0.2"
        self.info.icon_path = "icons/icon.png"
        self.info.authors = "Zhang, Chaoning and Han, Dongshen and Qiao, " \
                            "Yu and Kim, Jung Uk and Bae, Sung Ho and Lee, Seungkyu and Hong, Choong Seon"
        self.info.article = "Faster Segment Anything: Towards Lightweight SAM for Mobile Applications"
        self.info.journal = "arXiv"
        self.info.year = 2023
        self.info.license = "Apache License 2.0"
        # URL of documentation
        self.info.documentation_link = "https://arxiv.org/pdf/2306.14289.pdf"
        # Code source repository
        self.info.repository = "https://github.com/Ikomia-hub/infer_mobile_segment_anything"
        self.info.original_repository = "https://github.com/ChaoningZhang/MobileSAM"
        # Keywords used for search
        self.info.keywords = "Mobile SAM, ViT, Zero-Shot"
        self.info.algo_type = core.AlgoType.INFER
        self.info.algo_tasks = "SEMANTIC_SEGMENTATION"

    def create(self, param=None):
        # Create process object
        return InferMobileSegmentAnything(self.info.name, param)
