from datetime import datetime

class RipBox():
    def __init__(self, box, pval, timestamp, ind):
        self.box = box
        self.pval = pval
        self.timestamp = timestamp
        self.unique_id = '%s_%s' % (timestamp, ind)

    def __str__(self):
        return f"RipBox({self.boxes}, {self.pval}, {self.timestamp})"
    
    def get_timestamp(self):
        return self.timestamp
    
    def get_box(self):
        return self.box
    
    def get_pval(self):
        return self.pval
    
    def get_unique_id(self):
        return self.unique_id
    
    def get_center_coords(self):
        return (self.box[0] + self.box[2])/2, (self.box[1] + self.box[3])/2
    
    def get_width_height(self): 
        return self.box[2] - self.box[0], self.box[3] - self.box[1]
            
    def compute_iou(self, other):
        box_1 = self.get_box()
        box_2 = other.get_box()
        x1, y1, x2, y2 = box_1[0], box_1[1], box_1[2], box_1[3]
        x3, y3, x4, y4 = box_2[0], box_2[1], box_2[2], box_2[3]
        xA = max(x1, x3)
        yA = max(y1, y3)
        xB = min(x2, x4)
        yB = min(y2, y4)
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (x2 - x1 + 1) * (y2 - y1 + 1)
        boxBArea = (x4 - x3 + 1) * (y4 - y3 + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou
    
    def compute_area(self):
        return abs(self.box[2] - self.box[0]) * abs(self.box[3] - self.box[1])
    
    def compute_intersection(self, other):
        box_1 = self.get_box()
        box_2 = self.get_box()
        x1, y1, x2, y2 = box_1[0], box_1[1], box_1[2], box_1[3]
        x3, y3, x4, y4 = box_2[0], box_2[1], box_2[2], box_2[3]
        xA = max(x1, x3)
        yA = max(y1, y3)
        xB = min(x2, x4)
        yB = min(y2, y4)
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        return interArea
        
    def compute_bounding(self, other):
        return self.compute_intersection(other) / self.compute_area()
    
    def get_box_time_difference(self, other_box):
        timestamp1 = self.get_timestamp()
        timestamp2 = other_box.get_timestamp()
        # Define the format of the timestamps
        format = "%Y-%m-%d-%H-%M-%S"
        
        # Convert the timestamps to datetime objects
        time1 = datetime.strptime(timestamp1, format)
        time2 = datetime.strptime(timestamp2, format)
        
        # Calculate the difference in seconds
        difference = (time2 - time1).total_seconds()
        
        return difference