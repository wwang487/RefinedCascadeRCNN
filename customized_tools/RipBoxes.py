from .RipBox import RipBox
class RipBoxes():
    def __init__(self, box, pval, record_time, ind):
        self.boxes = [RipBox(box, pval, record_time, ind)]
        self.start_time = record_time
        self.end_time = record_time
        self.last_iou = 1
        self.last_bound = -1
        self.last_unique_id = self.boxes[0].get_unique_id()
        self.is_active = True
        self.is_main = True
        self.frame_num = 1

    def get_start_time(self):
        return self.start_time
    
    def get_end_time(self):
        return self.end_time
    
    def get_last_box(self):
        return self.boxes[-1]
    
    def get_first_box(self):
        return self.boxes[0]
    
    def get_boxes(self):
        return self.boxes
    
    def get_frame_num(self):
        return self.frame_num
    
    def get_last_iou(self):
        return self.last_iou
    
    def get_last_bound(self):
        return self.last_bound

    def get_last_unique_id(self):
        return self.last_unique_id
    
    def get_is_main(self):
        return self.is_main

    def get_second_last_box(self):
        return self.boxes[-2]

    def set_frame_num(self, frame_num):
        self.frame_num = frame_num

    def set_end_time(self, end_time):
        self.end_time = end_time
    
    def set_active(self, is_active):
        self.is_active = is_active
    
    def set_last_iou(self, last_iou):
        self.last_iou = last_iou
    
    def set_last_unique_id(self, last_unique_id):
        self.last_unique_id = last_unique_id

    def compute_mean_speed(self):
        first_x, first_y = self.get_first_box.get_center_coords()
        last_x, last_y = self.get_last_box.get_center_coords()
        return last_x - first_x, last_y - first_y
    
    def check_time_connectivity(self, other_boxes, time_thresh = 30):
        curr_last_box = self.get_last_box()
        other_first_box = other_boxes.get_first_box()
        return curr_last_box.get_box_time_difference(other_first_box) <= time_thresh
    
    def is_active(self):
        return self.is_active
    
    def add_box(self, box):
        self.boxes.append(box)
        self.end_time = box.get_timestamp()
        self.frame_num += 1

    def get_selected_boxes(self, rip_frame, iou_thresh = 0.2):
        max_iou, max_ind = -1, 0
        for i, box in enumerate(rip_frame.get_boxes()):
            iou = self.compute_iou(box)
            if iou > max_iou:
                max_iou = iou
                max_ind = i
        if max_iou >= iou_thresh:
            return rip_frame.get_box(max_ind)
        
    def pop_last_box(self):
        self.boxes.pop()
        self.end_time = self.boxes[-1].get_timestamp()
        self.frame_num -= 1
        self.is_active = False
        self.is_main = False
        self.last_iou = 1 if self.frame_num == 1 else self.get_last_box().compute_iou(self.boxes[-2])
        self.last_unique_id = self.boxes[-1].get_unique_id()

    def is_before_another_box(self, other_boxes):
        curr_last_box = self.get_last_box()
        other_first_box = other_boxes.get_first_box()
        return curr_last_box.get_box_time_difference(other_first_box) < 0
    
    def is_after_another_box(self, other_boxes):
        curr_first_box = self.get_first_box()
        other_last_box = other_boxes.get_last_box()
        return curr_first_box.get_box_time_difference(other_last_box) < 0
    
    def is_time_intersect(self, other_boxes, direction = 'ahead'):
        curr_first_box = self.get_first_box()
        curr_last_box = self.get_last_box()
        other_first_box = other_boxes.get_first_box()
        other_last_box = other_boxes.get_last_box()
        if direction == 'ahead':
            return curr_last_box.get_box_time_difference(other_last_box) < 0 and curr_last_box.get_box_time_difference(other_first_box) > 0
        elif direction == 'following':
            return curr_first_box.get_box_time_difference(other_last_box) < 0 and curr_first_box.get_box_time_difference(other_first_box) > 0
        
    def is_within_another_box(self, other_boxes):
        curr_first_box = self.get_first_box()
        curr_last_box = self.get_last_box()
        other_first_box = other_boxes.get_first_box()
        other_last_box = other_boxes.get_last_box()
        return curr_last_box.get_box_time_difference(other_last_box) < 0 and curr_first_box.get_box_time_difference(other_first_box) > 0