from xml.etree.ElementTree import ElementTree
import copy
from datetime import datetime

def compute_area(box):
    return abs(box[2] - box[0]) * abs(box[3] - box[1])

def compute_iou(box_1, box_2):
    x1, y1, x2, y2 = box_1
    x3, y3, x4, y4 = box_2
    xA = max(x1, x3)
    yA = max(y1, y3)
    xB = min(x2, x4)
    yB = min(y2, y4)
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (x2 - x1 + 1) * (y2 - y1 + 1)
    boxBArea = (x4 - x3 + 1) * (y4 - y3 + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def if_bound(box_1, box_2, thresh):
    tl_x1, tl_y1, br_x1, br_y1 = box_1
    tl_x2, tl_y2, br_x2, br_y2 = box_2
    area_box_2 = (br_x2 - tl_x2) * (br_y2 - tl_y2)
    overlap_x = max(0, min(br_x1, br_x2) - max(tl_x1, tl_x2))
    overlap_y = max(0, min(br_y1, br_y2) - max(tl_y1, tl_y2))
    overlap_area = overlap_x * overlap_y
    if area_box_2 == 0:
        percentage_within = 0
    else:
        percentage_within = overlap_area / area_box_2
    return percentage_within > thresh

def remove_bounded_obj(boxes, pvals, bound_thresh, p_low_thresh, p_high_thresh):
    # In this function, we will remove one box if it is within another box
    # The standard is: 1) if the ratio of one box is more than bound_thresh within another box
    # 2) exception: the p_val of the small box is very high > p_low_thresh
    #               and the p_val of the big box is very low < p_high_thresh
    to_delete = []
    
    if boxes == None or not boxes or not pvals:
        return boxes, pvals
    box_num = len(boxes)
    if box_num <= 1:
        return boxes, pvals
    else:
        for i in range(box_num):
            if i in to_delete:
                continue
            else:
                box_1 = boxes[i]
                for j in range(box_num):
                    if i == j or j in to_delete:
                        continue
                    else:
                        box_2 = boxes[j]
                        if if_bound(box_1, box_2, bound_thresh):
                            pval_1 = pvals[i]
                            pval_2 = pvals[j]
                            if not (pval_1 < p_low_thresh and pval_2 > p_high_thresh):
                                to_delete.append(j)
        if not to_delete:
            return boxes, pvals
        else:
            output_boxes, output_pvals = [], []
            for k in range(box_num):
                if k not in to_delete:
                    output_boxes.append(boxes[k])
                    output_pvals.append(pvals[k])
            return output_boxes, output_pvals
    
def remove_overlapped_obj(boxes, pvals, overlap_thresh):
    # In this function, we will remove the boxes if they are overlapped with each other
    to_delete = []
    box_num = len(boxes)
    if box_num <= 1:
        return boxes, pvals
    else:
        for i in range(box_num):
            if i in to_delete:
                continue
            else:
                box_1 = boxes[i]
                for j in range(i, box_num):
                    if i == j or j in to_delete:
                        continue
                    else:
                        box_2 = boxes[j]
                        iou_val = compute_iou(box_1, box_2)
                        if iou_val >= overlap_thresh:
                            pval_1 = pvals[i]
                            pval_2 = pvals[j]
                            if pval_1 >= pval_2:
                                to_delete.append(j)
                            else:
                                to_delete.append(i)
                                break
        if not to_delete:
            return boxes, pvals
        else:
            output_boxes, output_pvals = [], []
            for k in range(box_num):
                if k not in to_delete:
                    output_boxes.append(boxes[k])
                    output_pvals.append(pvals[k])
            return output_boxes, output_pvals

def remove_low_pval_obj(boxes, pvals, pval_thresh):
    if not boxes or not pvals:
        return boxes, pvals
    else:
        box_num = len(boxes)
        to_delete = []
        for i in range(box_num):
            if pvals[i] < pval_thresh:
                to_delete.append(i)
        if not to_delete:
            return boxes, pvals
        else:
            output_boxes, output_pvals = [], []
            for k in range(box_num):
                if k not in to_delete:
                    output_boxes.append(boxes[k])
                    output_pvals.append(pvals[k])
            return output_boxes, output_pvals

def remove_flat_objs(boxes, pvals, flat_thresh):
    # This function removes the boxes that are very flat.
    if not boxes or not pvals:
        return boxes, pvals
    else:
        box_num = len(boxes)
        to_delete = []
        for i in range(box_num):
            temp_box = boxes[i]
            ratio = (abs(temp_box[1] - temp_box[3]) + 0.00001) / (abs(temp_box[0] - temp_box[2]) + 0.00001)
            if ratio < flat_thresh:
                to_delete.append(i)
        if not to_delete:
            return boxes, pvals
        else:
            output_boxes, output_pvals = [], []
            for k in range(box_num):
                if k not in to_delete:
                    output_boxes.append(boxes[k])
                    output_pvals.append(pvals[k])
            return output_boxes, output_pvals

def remove_overlap_GT_obj(GT_boxes, overlap_thresh):
    
    to_delete = []
    box_num = len(GT_boxes)
    if box_num <= 1:
        return GT_boxes
    else:
        for i in range(box_num):
            if i in to_delete:
                continue
            else:
                box_1 = GT_boxes[i]
                for j in range(i, box_num):
                    if i == j or j in to_delete:
                        continue
                    else:
                        box_2 = GT_boxes[j]
                        iou_val = compute_iou(box_1, box_2)
                        if iou_val >= overlap_thresh:
                            area_1 = compute_area(box_1)
                            area_2 = compute_area(box_2)
                            if area_1 >= area_2:
                                to_delete.append(j)
                            else:
                                to_delete.append(i)
                                break
        if not to_delete:
            return GT_boxes
        else:
            output_boxes = []
            for k in range(box_num):
                if k not in to_delete:
                    output_boxes.append(GT_boxes[k])
            return output_boxes

def remove_bound_GT_boxes(GT_boxes, bound_thresh):
    pass

def remove_flat_GT_boxes(GT_boxes, flat_thresh):
    pass


        
def refine_boxes_for_single_timestamp(orig_detect_dict, overlap_thresh = 0.65, pval_thresh = 0.5, flat_thresh = 0.1,
                                      bound_thresh = 0.8, low_bound_p_thresh = 0.5, high_bound_p_thresh = 0.9,
                                      is_overlap_refine = True, is_bound_refine = True,
                                      is_lowpval_refine = True, is_flat_refine = False):
    timestamp_detect_dict = copy.deepcopy(orig_detect_dict)
    for key in timestamp_detect_dict.keys():
        temp_content = timestamp_detect_dict.get(key)
        boxes, pvals = temp_content.get('bboxes'), temp_content.get('pvals')
        if is_lowpval_refine:
            boxes, pvals = remove_low_pval_obj(boxes, pvals, pval_thresh)
        if is_bound_refine:
            boxes, pvals = remove_bounded_obj(boxes, pvals, bound_thresh, low_bound_p_thresh, high_bound_p_thresh)
        if is_overlap_refine:
            boxes, pvals = remove_overlapped_obj(boxes, pvals, overlap_thresh)
        if is_flat_refine:
            boxes, pvals = remove_flat_objs(boxes, pvals, flat_thresh)
        
        timestamp_detect_dict[key] = {"bboxes":boxes, 'pvals':pvals}
    return timestamp_detect_dict

def refine_boxes_for_single_img(temp_detect_dict, overlap_thresh = 0.65, pval_thresh = 0.5, flat_thresh = 0.1,
                                      bound_thresh = 0.8, low_bound_p_thresh = 0.5, high_bound_p_thresh = 0.9,
                                      is_overlap_refine = True, is_bound_refine = True,
                                      is_lowpval_refine = True, is_flat_refine = False):
    img_detect_dict = copy.deepcopy(temp_detect_dict)
    boxes, pvals = img_detect_dict.get('bboxes'), img_detect_dict.get('pvals')
    if is_lowpval_refine:
        boxes, pvals = remove_low_pval_obj(boxes, pvals, pval_thresh)
    if is_bound_refine:
        boxes, pvals = remove_bounded_obj(boxes, pvals, bound_thresh, low_bound_p_thresh, high_bound_p_thresh)
    if is_overlap_refine:
        boxes, pvals = remove_overlapped_obj(boxes, pvals, overlap_thresh)
    if is_flat_refine:
        boxes, pvals = remove_flat_objs(boxes, pvals, flat_thresh)
    img_detect_dict = {"bboxes":boxes, 'pvals':pvals}
    return img_detect_dict
        
def refine_boxes_for_multiple_datetimes(detect_dict, overlap_thresh = 0.65, pval_thresh = 0.5, flat_thresh = 0.1,
                                      bound_thresh = 0.8, low_bound_p_thresh = 0.5, high_bound_p_thresh = 0.9,
                                      choice = 'seperated',
                                      is_overlap_refine = True, is_bound_refine = True,
                                      is_lowpval_refine = True, is_flat_refine = False):
    # merged means already merging together to be one datetime, one image
    # separated means one datetime, multiple square images
    res_dict = {}
    for k in detect_dict.keys():
        temp_dict = detect_dict.get(k)
        if choice == 'merged':
            res_dict[k] = refine_boxes_for_single_img(temp_dict, overlap_thresh = overlap_thresh, pval_thresh = pval_thresh, flat_thresh = flat_thresh,
                                        bound_thresh = bound_thresh, low_bound_p_thresh = low_bound_p_thresh, high_bound_p_thresh = high_bound_p_thresh,
                                        is_overlap_refine = is_overlap_refine, is_bound_refine = is_bound_refine,
                                        is_lowpval_refine = is_lowpval_refine, is_flat_refine = is_flat_refine)  
        else:
            res_dict[k] = refine_boxes_for_single_timestamp(temp_dict, overlap_thresh = overlap_thresh, pval_thresh = pval_thresh, flat_thresh = flat_thresh,
                                      bound_thresh = bound_thresh, low_bound_p_thresh = low_bound_p_thresh, high_bound_p_thresh = high_bound_p_thresh,
                                      is_overlap_refine = is_overlap_refine, is_bound_refine = is_bound_refine,
                                      is_lowpval_refine = is_lowpval_refine, is_flat_refine = is_flat_refine)
    return res_dict
            



def sort_boxes_and_pvalues(boxes, pvalues, sort_by):
    """
    Sorts lists of boxes and corresponding p-values based on x1 or x2 coordinate of the boxes.

    Parameters:
        boxes (list of list): List containing sublists of box coordinates [x1, y1, x2, y2].
        pvalues (list): List containing p-values corresponding to each box.
        sort_by (int): Criterion for sorting; 1 for x1, 2 for x2.

    Returns:
        tuple: Tuple containing the sorted boxes and pvalues.
    """
    # Determine the index to sort by (0 for x1 and 2 for x2)
    index = 0 if sort_by == 1 else 2

    # Combine the boxes and pvalues for sorting
    combined = list(zip(boxes, pvalues))
    
    # Sort the combined list by the specified coordinate (x1 or x2)
    combined_sorted = sorted(combined, key=lambda x: x[0][index])
    
    # Unzip the sorted tuples back into separate lists
    sorted_boxes, sorted_pvalues = zip(*combined_sorted)
    
    return list(sorted_boxes), list(sorted_pvalues)

def merge_boxes_in_same_timestamp(input_dict, img_width, sensitivity_thresh = 3):
    # sensitivity thresh means the distance between the box boundary to the image edge.
    iter_keys = list(input_dict.keys())
    iter_keys.sort()
    res_boxes, res_pvals = [],[]
    new_x_coords = lambda x,ind,w: min(max(0, x + (ind - 1) * w), ind * w - 1)
    for key in iter_keys:
        if int(key) < len(iter_keys):
            if key == iter_keys[0]:
                curr_key = key
                curr_content = input_dict.get(curr_key)
                curr_boxes = curr_content.get('bboxes')
                curr_pvals = curr_content.get('pvals')
                curr_bools = [False] * len(curr_boxes) if len(curr_boxes) > 0 else []
            next_key = str(int(key) + 1)
            next_content = input_dict.get(next_key)
            next_boxes = next_content.get('bboxes')
            next_pvals = next_content.get('pvals')
            next_bools = [False] * len(next_boxes) if len(next_boxes) > 0 else []
            if not curr_bools:
                curr_bools, curr_key, curr_boxes, curr_pvals = next_bools, next_key, next_boxes, next_pvals
                continue
            else:
                if curr_bools[0]:
                    sorted_curr_boxes, sorted_curr_pvals = sort_boxes_and_pvalues(curr_boxes, curr_pvals, 1)
                    sorted_curr_boxes.pop(0)
                    sorted_curr_pvals.pop(0)
                    curr_boxes = sorted_curr_boxes
                    curr_pvals = sorted_curr_pvals
                    curr_bools.pop(0)
                if not curr_bools:
                    curr_bools, curr_key, curr_boxes, curr_pvals = next_bools, next_key, next_boxes, next_pvals
                    continue
                elif not next_bools:
                    for i in range(len(curr_boxes)):
                        temp_box, temp_p = curr_boxes[i], curr_pvals[i]
                        res_boxes.append([new_x_coords(temp_box[0], int(curr_key), img_width), temp_box[1], new_x_coords(temp_box[2], int(curr_key), img_width), temp_box[3]])
                        res_pvals.append(temp_p)
                    curr_bools, curr_key, curr_boxes, curr_pvals = next_bools, next_key, next_boxes, next_pvals
                else:
                    sorted_curr_boxes, sorted_curr_pvals = sort_boxes_and_pvalues(curr_boxes, curr_pvals, 2)
                    sorted_next_boxes, sorted_next_pvals = sort_boxes_and_pvalues(next_boxes, next_pvals, 1)
                    
                    if abs(sorted_curr_boxes[-1][2] - (img_width - 1)) >= sensitivity_thresh or abs(sorted_next_boxes[0][0]) >= sensitivity_thresh:
                        for i in range(len(sorted_curr_boxes)):
                            temp_box, temp_p = sorted_curr_boxes[i], sorted_curr_pvals[i]
                            res_boxes.append([new_x_coords(temp_box[0], int(curr_key), img_width), temp_box[1], new_x_coords(temp_box[2], int(curr_key), img_width), temp_box[3]])
                            res_pvals.append(temp_p)
                        curr_bools, curr_key, curr_boxes, curr_pvals = next_bools, next_key, next_boxes, next_pvals
                    else:
                        
                        for i in range(len(sorted_curr_boxes) - 1):
                            temp_box, temp_p = sorted_curr_boxes[i], sorted_curr_pvals[i]
                            res_boxes.append([new_x_coords(temp_box[0], int(curr_key), img_width), temp_box[1], new_x_coords(temp_box[2], int(curr_key), img_width), temp_box[3]])
                            res_pvals.append(temp_p)
                        
                        temp_curr_box, temp_curr_p = sorted_curr_boxes[-1], sorted_curr_pvals[-1]
                        temp_next_box, temp_next_p = sorted_next_boxes[0], sorted_next_pvals[0]
                        # print(temp_curr_box)
                        # print(temp_next_box)
                        new_y_min = min(temp_curr_box[1], temp_next_box[1])
                        new_y_max = max(temp_curr_box[3], temp_next_box[3])
                        new_x_min = new_x_coords(temp_curr_box[0], int(curr_key), img_width)
                        new_x_max = new_x_coords(temp_next_box[2], int(next_key), img_width)
                        new_pval = (temp_curr_p + temp_next_p) / 2
                        # print([new_x_min, new_y_min, new_x_max, new_y_max])
                        res_boxes.append([new_x_min, new_y_min, new_x_max, new_y_max])
                        res_pvals.append(new_pval)
                        next_bools[0] = True
                        curr_bools, curr_key, curr_boxes, curr_pvals = next_bools, next_key, next_boxes, next_pvals
        else:
            if not curr_bools:
                continue
            elif curr_bools[0] and len(curr_bools) == 1:
                continue
            else:
                if curr_bools[0]:
                    sorted_curr_boxes, sorted_curr_pvals = sort_boxes_and_pvalues(curr_boxes, curr_pvals, 1)
                    sorted_curr_boxes.pop(0)
                    sorted_curr_pvals.pop(0)
                    curr_boxes = sorted_curr_boxes
                    curr_pvals = sorted_curr_pvals
                for i in range(len(curr_boxes)):
                    temp_box, temp_p = curr_boxes[i], curr_pvals[i]
                    res_boxes.append([new_x_coords(temp_box[0], int(curr_key), img_width), temp_box[1], new_x_coords(temp_box[2], int(curr_key), img_width), temp_box[3]])
                    res_pvals.append(temp_p)
    return res_boxes, res_pvals

def merge_boxes_for_multiple_datetimes(input_dict, img_width, sensitivity_thresh = 3):
    res_dict = {}
    for key in input_dict.keys():
        temp_dict = input_dict.get(key)
        temp_boxes, temp_pvals = merge_boxes_in_same_timestamp(temp_dict, img_width, sensitivity_thresh = sensitivity_thresh)
        res_dict[key] = {"bboxes": temp_boxes, "pvals": temp_pvals}
    return res_dict

def compute_matching_for_single_img(GT_boxes, detect_boxes, iou_thresh = 0.5):
    # This function only works for single label type!!!!!!
    GT_num = len(GT_boxes)
    detect_num = len(detect_boxes)
    res_dict = {'TT':0, 'TF':0, 'FT':0}
    if GT_num == 0 and detect_num == 0:
        return res_dict
    elif GT_num == 0:
        res_dict['FT'] = detect_num
        return res_dict
    elif detect_num == 0:
        res_dict['TF'] = GT_num
        return res_dict
    else:
        matched = [False] * GT_num
        for i in range(detect_num):
            curr_detect_box = detect_boxes[i]
            max_ind, max_iou = 0, -1
            for j in range(GT_num):
                if not matched[j]:
                    temp_GT_box = GT_boxes[j]
                    temp_iou = compute_iou(curr_detect_box, temp_GT_box)
                    if temp_iou > max_iou:
                        max_iou = temp_iou
                        max_ind = j
            if max_iou >= iou_thresh:
                matched[max_ind] = True
                res_dict['TT'] = res_dict['TT'] + 1
            else:
                res_dict['FT'] = res_dict['FT'] + 1
        if sum(matched) != len(matched):
            res_dict['TF'] = res_dict['TF'] + len(matched) - sum(matched)
        return res_dict

def compute_matching_for_box_dict(GT_box_dict, detection_box_dict, iou_thresh = 0.5):
    res_dict = {}
    for timestamp in GT_box_dict.keys():
        timestamp_dict = {}
        temp_GT_dict = GT_box_dict.get(timestamp)
        temp_detection_dict = detection_box_dict.get(timestamp)
        for k in temp_GT_dict.keys():
            timestamp_dict[k] = compute_matching_for_single_img(temp_GT_dict.get(k), temp_detection_dict.get(k).get('bboxes'), iou_thresh = iou_thresh)
            
        res_dict[timestamp] = timestamp_dict
    return res_dict 
def compute_multilabel_matching_for_single_img(GT_boxes, GT_labels, detect_boxes, detect_labels, iou_thresh = 0.5):
    pass