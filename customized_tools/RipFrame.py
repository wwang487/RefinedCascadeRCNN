from .RipBox import RipBox
from .RipBoxes import RipBoxes
from datetime import datetime
import numpy as np

class RipFrame():
    def __init__(self, timestamp, boxes, pvals):
        self.timestamp = timestamp
        boxes_num = len(boxes)
        temp_boxes = []
        for i in range(boxes_num):
            #print(boxes[i])
            temp_boxes.append(RipBox(boxes[i], pvals[i], timestamp, i))
        self.timestamp = timestamp
        self.is_assigned = [False] * boxes_num
        self.boxes = temp_boxes

    def get_timestamp(self):
        return self.timestamp
    
    def get_boxes(self):
        return self.boxes
    
    def get_box(self, index):
        return self.boxes[index]
    
    def get_num_boxes(self):
        return len(self.boxes)
    
    def initialize_rip_boxes(self):
        active_rip_boxes = []
        for i in range(len(self.boxes)):
            active_rip_boxes.append(RipBoxes(self.boxes[i].get_box(), self.boxes[i].get_pval(), self.timestamp, i))
        return active_rip_boxes

    def update_active_rip_boxes(self, active_rip_boxes, iou_thresh = 0.2, bound_thresh = 0.8):
        new_complete_boxes, new_active_boxes = [], []
        for rip_boxes in active_rip_boxes:
            max_iou, max_ind = -1, 0
            last_box = rip_boxes.get_last_box()
            for i, box in enumerate(self.boxes):
                iou = last_box.compute_iou(box)
                if iou > max_iou:
                    max_iou = iou
                    max_ind = i
            if max_iou >= iou_thresh:
                rip_boxes.add_box(self.boxes[max_ind])
                self.is_assigned[max_ind] = True
                new_active_boxes.append(rip_boxes)
            else:
                # We also want to deal with the bounding case.
                # for i, box in enumerate(self.boxes):
                #     bounding = last_box.compute_bounding(box)
                #     if bounding > bound_thresh:
                #         rip_boxes.add_box(box)
                #         self.is_assigned[i] = True
                #         new_active_boxes.append(rip_boxes)
                #         break
                rip_boxes.set_active(False)
                new_complete_boxes.append(rip_boxes)
        for i in range(len(self.is_assigned)):
            if not self.is_assigned[i]:
                new_active_boxes.append(RipBoxes(self.boxes[i].get_box(), self.boxes[i].get_pval(), self.timestamp, i))
        return new_active_boxes, new_complete_boxes
    
    def get_max_dist(self):
        y_coords = [box.get_box()[3] for box in self.boxes]
        x1_positions = [box.get_box()[0] for box in self.boxes]
        x2_positions = [box.get_box()[2] for box in self.boxes]
        y_max_ind = np.argmax(y_coords)
        return y_coords[y_max_ind], (x1_positions[y_max_ind] + x2_positions[y_max_ind]) / 2
    
    def get_frame_time_difference(self, timestamp2):
        timestamp1 = self.get_timestamp()
        # Define the format of the timestamps
        format = "%Y-%m-%d-%H-%M-%S"
        
        # Convert the timestamps to datetime objects
        time1 = datetime.strptime(timestamp1, format)
        time2 = datetime.strptime(timestamp2, format)
        
        # Calculate the difference in seconds
        difference = (time2 - time1).total_seconds()
        
        return difference