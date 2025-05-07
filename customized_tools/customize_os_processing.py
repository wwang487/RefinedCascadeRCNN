import pickle
import os
import json
from xml.etree.ElementTree import ElementTree
import os
import argparse
import pandas as pd
import random
from shutil import copyfile
import xml.dom.minidom
import xml.etree.ElementTree as ET
from tqdm import tqdm
import cv2
import shutil
from typing import Dict, List, Tuple
from collections import defaultdict

def read_xml(in_path):
    tree = ElementTree()
    tree.parse(in_path)
    return tree

def read_vals_within_xml_file(file_folder, file_path):
    tree = read_xml(file_folder + file_path)
    root = tree.getroot()
    img_width = int(root.find('size/width').text)
    img_height = int(root.find('size/height').text)
    res = {}
    for child in root.findall('object'):
        bndbox = child.find('bndbox')
        spe = child.find('name').text
        xmin = int(bndbox[0].text)
        ymin = int(bndbox[1].text)
        xmax = int(bndbox[2].text)
        ymax = int(bndbox[3].text)
        res.append({spe:[xmin,ymin,xmax,ymax]})
    return img_width, img_height, res

def parse_xml(file_path: str) -> Tuple[List[Dict], int]:
    """
    Parse an XML file and extract bounding box information and the width of the image.
    """
    tree = ET.parse(file_path)
    root = tree.getroot()
    boxes = []
    
    for obj in root.iter('object'):
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        boxes.append([xmin, ymin, xmax, ymax])

    return boxes

def process_xml_folder(folder_path):
    """
    Process all XML files in a folder and merge boxes based on timestamp.
    """
    files = os.listdir(folder_path)
    res_dict = {}

    for file in sorted(files):
        if file.endswith('.xml'):
            
            timestamp = '-'.join(file.split('-')[:6])
            ind = file.split('.')[0].split('_')[-1]
            boxes = parse_xml(os.path.join(folder_path, file))
            if not res_dict or timestamp not in res_dict:
                timestamp_dict = {ind:boxes}
            else:
                timestamp_dict = res_dict.get(timestamp)
                timestamp_dict[ind] = boxes
                
            res_dict[timestamp] = timestamp_dict

    return res_dict

def process_xml_folder_with_list(folder_path, xml_list):
    """
    Process all XML files in a folder and merge boxes based on timestamp.
    """
    files = os.listdir(folder_path)
    res_dict = {}

    for xl in xml_list:
        file = xl + '.xml'
        timestamp = '-'.join(file.split('-')[:6])
        ind = file.split('.')[0].split('_')[-1]
        boxes = parse_xml(os.path.join(folder_path, file))
        if not res_dict or timestamp not in res_dict:
            timestamp_dict = {ind:boxes}
        else:
            timestamp_dict = res_dict.get(timestamp)
            timestamp_dict[ind] = boxes
                
        res_dict[timestamp] = timestamp_dict

    return res_dict

def load_pickle_data(pickle_file):
    # for reading also binary mode is important
    dbfile = open(pickle_file, 'rb')    
    db = pickle.load(dbfile)
    dbfile.close()
    return db

def dump_pickle_data(pickle_file, content):
    # for reading also binary mode is important
    dbfile = open(pickle_file, 'wb')    
    pickle.dump(content, dbfile)
    dbfile.close()

def convert_orig_pkl_to_dict(orig_list):
    res = {}
    for i in orig_list:
        temp_img = i.get('img_id')
        temp_detection = i.get('pred_instances')
        temp_datetime = '-'.join(temp_img.split('-')[:-1])
        temp_id = temp_img.split('.')[0].split('_')[-1]
        temp_boxes = temp_detection.get('bboxes').tolist()
        temp_score = temp_detection.get('scores').tolist()
        if temp_datetime not in res:
            res[temp_datetime] = {temp_id: {'bboxes': temp_boxes, 'pvals': temp_score}}
        else:
            temp_dict = res.get(temp_datetime)
            temp_dict[temp_id] = {'bboxes': temp_boxes, 'pvals': temp_score}
            res[temp_datetime] = temp_dict
    return res

def get_all_files_with_suffix(directory, suffix):
    # List all files in the given directory
    files = os.listdir(directory)
    # Filter files that end with the given suffix
    matched_files = [file for file in files if file.endswith(suffix)]
    return matched_files

def get_all_filefront_with_marker(mark_folder, mark_suffix):
    all_imgs = []
    for file in os.listdir(mark_folder):
        if (file[-3:] == mark_suffix):
            all_imgs.append(file.split('.')[0])
    all_imgs = pd.DataFrame(all_imgs, columns=["file"])
    return all_imgs

def read_json_file(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    file.close()
    return data

def construct_detection_df_for_single_day(parent_folder_path, date_str, pred_folder_name, data_suffix = 'json'):
    json_folder = '%s/%s/%s/'%(parent_folder_path, date_str, pred_folder_name)
    json_files = get_all_files_with_suffix(json_folder, data_suffix)
    res_dict = {}
    for jf in json_files:
        time_str = '-'.join(jf.split('_')[0].split('-')[:-1])
        number_str = jf.split('_')[-1].split('.')[0]
        temp_detection_res = read_json_file(json_folder + jf)
        if time_str not in res_dict:
            temp_date_dict = {number_str:{'bboxes': temp_detection_res.get('bboxes'), 
                                          'pvals': temp_detection_res.get('scores')}}
        else:
            temp_date_dict = res_dict.get(time_str)
            temp_date_dict[number_str] = {'bboxes': temp_detection_res.get('bboxes'), 
                                          'pvals': temp_detection_res.get('scores')}
        res_dict[time_str] = temp_date_dict
    return res_dict

def divide_imgs_into_different_sets(all_imgs):
    img_names = list(all_imgs.file)
    img_num = len(img_names)
    
    the_list = list(range(0, img_num))
    
    random.shuffle(the_list)
    trainval = []
    train = []
    val = []
    test = []
    
    
    for i in range(img_num):
        if i <= int(img_num * 0.7):
            trainval.append(img_names[the_list[i]])
            if i <= int(img_num * 0.35):
                train.append(img_names[the_list[i]])
            else:
                val.append(img_names[the_list[i]])
        else:
            test.append(img_names[i])
    return trainval, train, val, test

def copy_file_list_to_new_folder(tar_list, old_folder, new_folder):
    os.makedirs(new_folder, exist_ok = True)
    for i in tar_list:
        old_path = old_folder + i
        new_path = new_folder + i
        copyfile(old_path, new_path)

def copy_XML_file_list_to_new_folder(tar_list, old_folder, new_folder):
    os.makedirs(new_folder, exist_ok = True)
    print('Start Copying Marking Files')
    for i in tqdm(tar_list):
        old_path = old_folder + i.split('.')[0] + '.xml'
        new_path = new_folder + i.split('.')[0] + '.xml'
        copyfile(old_path, new_path)
        
def copy_JPG_file_list_to_new_folder(tar_list, old_folder, new_folder):
    os.makedirs(new_folder, exist_ok = True)
    print('Start Copying Image Files')
    for i in tqdm(tar_list):
        old_path = old_folder + i.split('.')[0] + '.jpg'
        new_path = new_folder + i.split('.')[0] + '.jpg'
        copyfile(old_path, new_path)
        
def write_train_val_test_file(cand_list, save_folder, save_name):
    os.makedirs(save_folder, exist_ok = True)
    with open(save_folder + save_name, 'w') as f1:
        trainval_len = len(cand_list)
        for i in range(trainval_len):
            if i == trainval_len - 1:
                temp_img = cand_list[i]
                f1.write(temp_img.split('.')[0])
            else:
                temp_img = cand_list[i]
                f1.write(temp_img.split('.')[0])
                f1.write('\n')
        f1.close()

def modify_xml_files(input_marker_folder, input_img_folder,
                     output_marker_folder, output_img_folder = '', output_img_choice = False):
    # If the input image is not jpg, you also need to convert them into jpg by
    # setting output_img_choice to True and specify the output_img_folder
    if not os.path.exists(output_marker_folder):
        os.makedirs(output_marker_folder, exist_ok = True)
    if output_img_choice:
        if not os.path.exists(output_img_folder):
            os.makedirs(output_img_folder, exist_ok = True)
    files = os.listdir(input_img_folder)  #get all files within a folder

    for f in tqdm(files):
        
        file_front = f.split('.')[0]
        xml_file_name = file_front + '.xml'
        
        
        if os.path.exists(input_marker_folder + xml_file_name):
            if output_img_choice:
                curr_img = cv2.imread(input_img_folder + f)
                new_img_name = file_front + '.' + 'jpg'
                cv2.imwrite(output_img_folder + new_img_name, curr_img)
            
            dom = xml.dom.minidom.parse(os.path.join(input_marker_folder, xml_file_name)) 
            root = dom.documentElement
            name = root.getElementsByTagName('name')
            folder = root.getElementsByTagName('folder')
            paths = root.getElementsByTagName('path')
            
            for i in range(len(folder)):  
                #print (folder[i].firstChild.data)
                folder[i].firstChild.data='VOC2007'
                #print (folder[i].firstChild.data)
                
            for j in range(len(paths)):
                paths[j].firstChild.data = '.'.join([paths[j].firstChild.data.split('.')[0],'jpg'])
                
            with open(os.path.join(output_marker_folder, xml_file_name),'w') as fh:
                dom.writexml(fh)

def clear_folder(folder_path):
    """
    Removes all the contents of a specified folder.
    
    Parameters:
        folder_path (str): Path to the folder to clear.
    """
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.remove(item_path)  # Remove files and links
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)  # Remove directories
    print(f"All contents removed from {folder_path}")

def prepare_voc_dataset(input_marker_folder, input_img_folder, save_folder, mark_suffix = 'xml'):
    if os.path.exists(save_folder):
        clear_folder(save_folder)
    save_div_folder = save_folder + 'ImageSets/Main/'
    os.makedirs(save_div_folder, exist_ok = True)

    save_anno_folder = save_folder + 'Annotations/'
    os.makedirs(save_anno_folder, exist_ok = True)

    save_img_folder = save_folder + 'JPEGImages/'
    os.makedirs(save_img_folder, exist_ok = True)

    all_imgs = get_all_filefront_with_marker(input_marker_folder, mark_suffix)

    copy_XML_file_list_to_new_folder(list(all_imgs.file), input_marker_folder, save_anno_folder)
    copy_JPG_file_list_to_new_folder(list(all_imgs.file), input_img_folder, save_img_folder)

    trainval, train, val, test = divide_imgs_into_different_sets(all_imgs)

    trainval_name = 'trainval.txt'
    train_name = 'train.txt'
    val_name = 'val.txt'
    test_name =  'test.txt'

    write_train_val_test_file(trainval, save_div_folder, trainval_name)
    write_train_val_test_file(train, save_div_folder, train_name)
    write_train_val_test_file(val, save_div_folder, val_name)
    write_train_val_test_file(test, save_div_folder, test_name)