from .RipBoxes import RipBoxes
from .RipBox import RipBox
from .RipFrame import RipFrame

from datetime import datetime
import numpy as np

def track_rip_boxes(rip_dict):
    keys = list(rip_dict.keys())
    keys.sort()
    clean_complete_boxes = []
    for k in range(len(keys)):
        # print(k)
        temp_content = rip_dict.get(keys[k])
        rip_frame = RipFrame(str(keys[k]), temp_content.get('bboxes'), temp_content.get('pvals'))
        if k == 0:
            active_rip_boxes = rip_frame.initialize_rip_boxes()
        else:
            active_rip_boxes, temp_clean_complete_boxes = rip_frame.update_active_rip_boxes(active_rip_boxes)
            clean_complete_boxes += temp_clean_complete_boxes
            active_rip_boxes, temp_clean_complete_boxes = split_merging(active_rip_boxes)
            clean_complete_boxes += temp_clean_complete_boxes
    return clean_complete_boxes

def split_merging(active_rip_boxes):
    check_num = len(active_rip_boxes)
    clean_active_boxes, clean_complete_boxes = [], []
    to_clean = []
    for i in range(check_num):
        if active_rip_boxes[i].get_frame_num() == 1 or i in to_clean:
            continue
        else:
            last_check_id = active_rip_boxes[i].get_last_box().get_unique_id()
            last_iou = active_rip_boxes[i].get_last_iou()
            for j in range(i + 1, check_num):
                if active_rip_boxes[j].get_frame_num() == 1 or j in to_clean:
                    continue
                if active_rip_boxes[j].get_last_box().get_unique_id() == last_check_id:
                    temp_iou = active_rip_boxes[j].get_last_iou()
                    if temp_iou > last_iou:
                        to_clean.append(j)
                    else:
                        to_clean.append(i)
    for i in range(check_num):
        if i not in to_clean:
            clean_active_boxes.append(active_rip_boxes[i])
        else:
            temp_boxes = active_rip_boxes[i]
            temp_boxes.pop_last_box()
            clean_complete_boxes.append(temp_boxes)
    return clean_active_boxes, clean_complete_boxes

def filter_main_boxes(final_boxes):
    res_boxes = []
    for f_b in final_boxes:
        if f_b.get_is_main():
            res_boxes.append(f_b)
    return res_boxes

def eliminate_short_rips(final_boxes, thresh = 2):
    res_boxes = []
    for f_b in final_boxes:
        if f_b.get_frame_num() > thresh:
            res_boxes.append(f_b)
    return res_boxes

def get_duration(timestamp1, timestamp2):
    
    format = "%Y-%m-%d-%H-%M-%S"
        
    # Convert the timestamps to datetime objects
    time1 = datetime.strptime(timestamp1, format)
    time2 = datetime.strptime(timestamp2, format)
        
    # Calculate the difference in seconds
    difference = (time2 - time1).total_seconds()
        
    return abs(difference)

def connect_active_boxes(active_rip_boxes):
    pass

def split_events_based_on_max_dist(rip_dict, time_thresh = 180, duration_thresh = 30, peak_period_thresh = 5, dist_percent_thresh = 0.5, max_dist_thresh = 50):
    # if the time difference is larger than a threshold, if the rip current max distance changes a lot, then it's a new event.
    keys = list(rip_dict.keys())
    keys.sort()
    final_events = []
    temp_event = {}
    temp_max_dist = -1
    for i in range(len(keys)):
        temp_timestamp = str(keys[i])
        temp_content = rip_dict.get(temp_timestamp)
        temp_boxes = temp_content.get('bboxes')
        temp_pvals = temp_content.get('pvals')
        temp_box_num = len(temp_boxes)
        if temp_box_num == 0:
            continue
        temp_rip_frame = RipFrame(temp_timestamp, temp_boxes, temp_pvals)
        temp_dist, temp_xcenter = temp_rip_frame.get_max_dist()
        if temp_max_dist == -1:
            # If the first event frame, initialize one
            if len(final_events) == 0:
                temp_max_dist = temp_dist
                temp_event['start_dist'] = temp_dist
                temp_event['max_dist'] = temp_max_dist
                temp_event['end_dist'] = temp_max_dist
                temp_event['end_time'] = temp_timestamp
                temp_event['start_time'] = temp_timestamp
                temp_event['peak_time'] = temp_timestamp
                temp_event['max_x_center'] = temp_xcenter
                temp_event['highest_box_num'] = temp_box_num
                temp_event['max_boxes'] = temp_boxes
            # If not the first event frame, compare current distance with the end distance of the last detected event,
            # if the distance is smaller than the last distance and the time difference with the ending time of last event is smaller than the thresh,
            # this frame is still in a decaying process of the last event, so we should continue.
            elif temp_dist < final_events[-1].get('end_dist') and abs(temp_rip_frame.get_frame_time_difference(final_events[-1].get('end_time'))) < time_thresh:
                continue
            # Otherwise, means either it is bouncing back or longer than the pre-set cooling time, should be considered as a new event.
            else:
                temp_max_dist = temp_dist
                temp_event['start_dist'] = temp_dist
                temp_event['max_dist'] = temp_max_dist
                temp_event['end_dist'] = temp_max_dist
                temp_event['end_time'] = temp_timestamp
                temp_event['start_time'] = temp_timestamp
                temp_event['peak_time'] = temp_timestamp
                temp_event['max_x_center'] = temp_xcenter
                temp_event['highest_box_num'] = temp_box_num
                temp_event['max_boxes'] = temp_boxes
        else:
            # If previous frame is within a event, we need to do the following:
            # If the time difference between current timestamp and the ending time of current event is longer than the time thresh, means 
            # the current timestamp already exceeds the decaying time threshold as a separate rip event.
            if abs(temp_rip_frame.get_frame_time_difference(temp_event.get('end_time'))) > time_thresh:
                temp_event['duration'] = get_duration(temp_event.get('end_time'), temp_event.get('start_time'))
                temp_event['peak_time_cost'] = get_duration(temp_event.get('peak_time'), temp_event.get('start_time'))
                if temp_event.get('duration') >= duration_thresh and temp_event.get('max_dist') >= max_dist_thresh and temp_event.get('peak_time_cost') > peak_period_thresh:
                    if temp_event.get('peak_time_cost') > 0:
                        temp_event['speed'] = (temp_event.get('max_dist') - temp_event.get('start_dist')) / temp_event.get('peak_time_cost')
                    else:
                        temp_event['speed'] = np.nan
                    final_events.append(temp_event)
                temp_max_dist = -1
                temp_event = {}
            else:
                # If we found a frame with more box numbers, update
                if temp_event.get('highest_box_num') < temp_box_num:
                    temp_event['highest_box_num'] = temp_box_num
                # If the distance is higher than the current max, we need to update
                if temp_dist > temp_event.get('max_dist'):
                    temp_event['peak_time'] = temp_timestamp
                    temp_event['max_dist'] = temp_dist
                    temp_event['end_dist'] = temp_dist
                    temp_event['end_time'] = temp_timestamp
                    temp_event['max_x_center'] = temp_xcenter
                    temp_event['max_boxes'] = temp_boxes

                # If the distance is much smaller than the current max, means the flash rip events end.
                elif temp_dist < temp_event.get('max_dist') * dist_percent_thresh:
                    temp_event['duration'] = get_duration(temp_event.get('end_time'), temp_event.get('start_time'))
                    temp_event['peak_time_cost'] = get_duration(temp_event.get('peak_time'), temp_event.get('start_time'))
                    if temp_event.get('duration') >= duration_thresh and temp_event.get('max_dist') >= max_dist_thresh and temp_event.get('peak_time_cost') > peak_period_thresh:
                        if temp_event.get('peak_time_cost') > 0:
                            temp_event['speed'] = (temp_event.get('max_dist') - temp_event.get('start_dist')) / temp_event.get('peak_time_cost')
                        else:
                            temp_event['speed'] = np.nan
                        final_events.append(temp_event)
                    temp_max_dist = -1
                    temp_event = {}
                # Otherwise, just update the current distance and timestamp.
                else:
                    temp_event['end_dist'] = temp_dist
                    temp_event['end_time'] = temp_timestamp
    return final_events         