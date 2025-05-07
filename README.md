# RefinedCascadeRCNN
This is the code for refining cascade RCNN outputs

First, follow mmdetection developed by openmmlab, URL: https://github.com/open-mmlab/mmdetection to prepare your dataset in VOC format, and then train with cascade R-CNN, or any other object detection models.

Then, if you want to refine a single image, you can call the function refine_boxes_for_single_img written in customized_tools/customize_box_refine.py
 
And if you want to refine a single timestamp that contains multiple images, you can call the function refine_boxes_for_single_timestamp written in customized_tools/customize_box_refine.py

And if you want to merge the square image back to the ortho image, and merge the adjacent boxes, you can call the function merge_boxes_in_same_timestamp written in customized_tools/customize_box_refine.py
