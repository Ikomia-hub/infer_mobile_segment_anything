<div align="center">
  <img src="https://raw.githubusercontent.com/Ikomia-hub/infer_mobile_segment_anything/main/icons/meta_icon.jpg" alt="Algorithm icon">
  <h1 align="center">infer_mobile_segment_anything</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/infer_mobile_segment_anything">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/infer_mobile_segment_anything">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/infer_mobile_segment_anything/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/infer_mobile_segment_anything.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>

This algorithm proposes inference for the Mobile Segment Anything Model (MobileSAM). It can be used to generate masks for all objects in an image. With its promptable segmentation capability, SAM delivers unmatched versatility for various image analysis tasks. 

![Sam dog](https://raw.githubusercontent.com/Ikomia-hub/infer_mobile_segment_anything/main/output.jpg)


## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow

[Change the sample image URL to fit algorithm purpose]

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add algorithm
algo  = wf.add_task(name = "infer_mobile_segment_anything", auto_connect=True)

# Run directly on your image
wf.run_on(url="https://raw.githubusercontent.com/Ikomia-dev/notebooks/main/examples/img/img_cat.jpg")

# Inspect your result
display(algo.get_image_with_mask())
```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.

- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).

- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).

## :pencil: Set algorithm parameters


- **input_box** (list): A Nx4 array of given box prompts to the  model, in [XYXY] or [[XYXY], [XYXY]] format.
- **draw_graphic_input** (Boolean): When set to True, it allows you to draw graphics (box or point) over the object you wish to segment. If set to False, SAM will automatically generate masks for the entire image.
- **points_per_side** (int or None, *optional*): The number of points to be sampled for mask generation when running automatic segmentation.
- **input_point** (list, *optional*): A Nx2 array of point prompts to the model. Each point is in [X,Y] in pixels.
- **input_point_label** (list, *optional*): A length N array of labels for the point prompts. 1 indicates a foreground point and 0 indicates a background point
- **points_per_side** (int) - default '32' : (Automatic detection mode). The number of points to be sampled along one side of the image. The total number of points is points_per_side**2. 
- **points_per_batch** (int) - default '64': (Automatic detection mode).  Sets the number of points run simultaneously by the model. Higher numbers may be faster but use more GPU memory.
- **stability_score_thresh** (float) - default '0.95': Filtering threshold in [0,1], using the stability of the mask under changes to the cutoff used to binarize the model's mask predictions.
- **box_nms_thresh** (float) - default '0.7': The box IoU cutoff used by non-maximal suppression to filter duplicate masks.
- **iou_thres** (float) - default '0.88': A filtering threshold in [0,1], using the model's predicted mask quality.
- **crop_n_layers** (int) - default '0' : If >0, mask prediction will be run again oncrops of the image. Sets the number of layers to run, where each layer has 2**i_layer number of image crops.
- **crop_nms_thresh** (float) - default '0': The box IoU cutoff used by non-maximal suppression to filter duplicate masks between different crops.
- **crop_overlap_ratio** (float) default 'float(512 / 1500)'
- **crop_n_points_downscale_factor** (int) - default '1' : The number of points-per-side sampled in layer n is scaled down by crop_n_points_downscale_factor**n.
- **min_mask_region_area** (int) - default '0': op layer. Exclusive with points_per_side. min_mask_region_area (int): If >0, postprocessing will be applied to remove disconnected regions and holes in masks with area smaller than min_mask_region_area. 
- **input_size_percent** (int) - default '100': Percentage size of the input image. Can be reduce to save memory usage. 



```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display


# Init your workflow
wf = Workflow()

# Add algorithm
algo  = wf.add_task(name = "infer_mobile_segment_anything", auto_connect=True)

algo.set_parameters({
    "draw_graphic_input": "False",
    "points_per_side": "16",
    "iou_thres": "0.88",
    "input_size_percent": "100",
})

# Run directly on your image
wf.run_on(url="https://raw.githubusercontent.com/Ikomia-dev/notebooks/main/examples/img/img_cat.jpg")

# Inspect your result
display(algo.get_image_with_mask())
```

## :mag: Explore algorithm outputs

Every algorithm produces specific outputs, yet they can be explored them the same way using the Ikomia API. For a more in-depth understanding of managing algorithm outputs, please refer to the [documentation](https://ikomia-dev.github.io/python-api-documentation/advanced_guide/IO_management.html).

```python
import ikomia
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_mobile_segment_anything", auto_connect=True)

# Run on your image  
wf.run_on(url="https://raw.githubusercontent.com/Ikomia-dev/notebooks/main/examples/img/img_cat.jpg")

# Iterate over outputs
for output in algo.get_outputs():
    # Print information
    print(output)
    # Export it to JSON
    output.to_json()
```

## :fast_forward: Advanced usage 

## 1. Automated mask generation
When no prompt is used, MobileSAM will generate masks automatically over the entire image. 
You can select the number of masks using the parameter "Points per side" on Ikomia STUDIO or "points_per_side" with the API. Here is an example using the default settings (32 points/side).  

![Sam dog auto](https://raw.githubusercontent.com/Ikomia-hub/infer_segment_anything/main/images/dog_auto_seg.png)


## 2. Segmentation mask with graphic prompts:
Given a graphic prompts: a single point or boxes SAM can predict masks over the desired objects. 
- Ikomia API: 
    - Using graphics: Set the parameter draw_graphic_input=True to draw over the image.
        - Point: A point can be generated with a left click
        - Box: Left click > drag > release
    - Using prompt coordinate
        - Point: 'input_point' parameter, e.g. [xy]
        - Point label: 'input_point_label' parameters, e.g. [1,0] 1 to include, 0 to exclude from mask
        - Box: 'input_box' parameter, e,g, [xyxy] or [[xyxy], [xyxy]].


- Ikomia STUDIO:
    - Using graphics
        - Point: Select the point tool
        - Box: Select the Square/Rectangle tool
    - Using coordinate prompts
        - Point: 'Point coord. xy (optional)' [xy]
        - Point label: [1,0], 1 to include, 0 to exclude from mask
        - Box: 'Box coord. xyxy (optional)' [[xyxy], [xyxy]]

### a. Single point 
SAM with generate three outputs given a single point (3 best scores). 
You can select which mask to output using the mask_id parameters (1, 2 or 3) 

![Sam dog single](https://raw.githubusercontent.com/Ikomia-hub/infer_segment_anything/main/images/dog_single_point.png)


### b. Multiple points
A single point can be ambiguous, using multiple points can improve the quality of the expected mask.

### c. Boxes
Drawing a box over the desired object usually output a mask closer to expectation compared to point(s). 

SAM can also take multiple inputs prompts.
![Sam cat boxes](https://raw.githubusercontent.com/Ikomia-hub/infer_segment_anything/main/images/cats_boxes.png)

### d. Point and box

Point and box can be combined by including both types of prompts to the predictor. Here this can be used to select just the trucks's tire, instead of the entire wheel.
![truck_box_point](https://raw.githubusercontent.com/Ikomia-hub/infer_segment_anything/main/images/truck_box_point.png)














































The Segment Anything Model (SAM) offers multiple inference modes for generating masks:
1. Automated mask generation (segmentation over the full image)
2. Segmentation masks from prompts (bounding boxes or point)

The MobileSAM model is an adaptation of SAM keeping the same pipeline as the original SAM. The lightweight MobileSAM has been trained with 100k datasets (1% of the original images used for SAM). MobileSAM is 60 times smalled then SAM ViT-H yet performs on par with the original SAM.


## 1. Automated mask generation
When no prompt is used, MobileSAM will generate masks automatically over the entire image. 
You can select the number of masks using the parameter "Points per side" on Ikomia STUDIO or "points_per_side" with the API. Here is an example with ViT-H using the default settings (32 points/side).

<img src="https://github.com/Ikomia-hub/infer_mobile_segment_anything/blob/main/images/dog_auto_seg.png"  width="30%" height="30%">

## 2. Segmentation mask with graphic prompts:
Given a graphic prompts: a single point or boxes MobileSAM can predict masks over the desired objects. 
- Ikomia API: Add the parameter "image_path = 'PATH/TO/YOUR/IMAGE'"  to draw over the image.
    - Point: A point can be generated with a left click
    - Box: Left click > drag > release

- Ikomia STUDIO: Open the Toggle graphics toolbar 
    - Point: Select the point tool
    - Box: Select the Square/Rectangle tool

### a. Single point 
SAM with generate three outputs given a single point (3 best scores). 
You can select which mask to output using the mask_id parameters (1, 2 or 3) 
<img src="https://github.com/Ikomia-hub/infer_mobile_segment_anything/blob/main/images/dog_single_point.png"  width="80%" height="80%">

### b. Boxes
A single point can be ambiguous, drawing a box over the desired object usually output a mask closer to expectation. 

SAM can also take multiple inputs prompts.

<img src="https://github.com/Ikomia-hub/infer_mobile_segment_anything/blob/main/images/cats_boxes.png"  width="80%" height="80%">

### c. Point and box

Point and box can be combined by including both types of prompts to the predictor. Here this can be used to select just the trucks's tire, instead of the entire wheel.

<img src="https://github.com/Ikomia-hub/infer_mobile_segment_anything/blob/main/images/truck_box_point.png"  width="80%" height="80%">


## Inference with Ikomia API

``` python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()    

# Add the MobileSAM algorithm to your workflow
mobilesam = wf.add_task(name="infer_mobile_segment_anything", auto_connect=True)

# To use graphics prompts, you need to set the image path parameter
# (Local image only, no url) 
# mobilesam.set_parameters({ 
#     "image_path": "C:/PATH/TO/YOUR/IMAGE",
# }) 

# Run on your image  
wf.run_on(url="https://raw.githubusercontent.com/Ikomia-dev/notebooks/main/examples/img/img_dog.png")

# Inspect your results
display(mobilesam.get_image_with_mask())
```

