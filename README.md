# infer_segment_anything

The Segment Anything Model (SAM) offers multiple inference modes for generating masks:
1. Automated mask generation (segmentation over the full image)
2. Segmentation masks from prompts (bounding boxes or point)

The MobileSAM model is an adaptation of SAM keeping the same pipeline as the original SAM. The lightweight MobileSAM has been trained with 100k datasets (1% of the original images used for SAM). MobileSAM is 60 times smalled then SAM ViT-H yet performs on par with the original SAM.


## 1. Automated mask generation
When no prompt is used, MobileSAM will generate masks automatically over the entire image. 
You can select the number of masks using the parameter "Points per side" on Ikomia STUDIO or "points_per_side" with the API. Here is an example with ViT-H using the default settings (32 points/side).  

<img scr="https://github.com/Ikomia-hub/infer_segment_anything/blob/main/images/dog_auto_seg.png"  width="30%" height="30%">


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
<img src="https://github.com/Ikomia-hub/infer_segment_anything/blob/main/images/dog_single_point.png"  width="80%" height="80%">

### b. Boxes
A single point can be ambiguous, drawing a box over the desired object usually output a mask closer to expectation. 

SAM can also take multiple inputs prompts.

<img src="https://github.com/Ikomia-hub/infer_segment_anything/blob/main/images/cats_boxes.png"  width="80%" height="80%">

### c. Point and box

Point and box can be combined by including both types of prompts to the predictor. Here this can be used to select just the trucks's tire, instead of the entire wheel.

<img src="https://github.com/Ikomia-hub/infer_segment_anything/blob/main/images/truck_box_point.png"  width="80%" height="80%">


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

