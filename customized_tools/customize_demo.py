from mmengine.logging import print_log
from mmdet.apis import DetInferencer
import os

def create_demo_parse(input_img_path, input_model_path, input_weight_path,\
                    output_path, if_show, if_save_vis, if_save_pred, \
                    if_print_res, device = 'cuda:0', palette = 'voc', batch_size = 1):
    os.makedirs(output_path, exist_ok = True)
    call_args = {'inputs': input_img_path, 'model' : input_model_path, 'weights' : input_weight_path,\
        'out_dir': output_path, 'device' : device, 'show': if_show, 'no_save_vis' : not if_save_vis, \
        'no_save_pred': not if_save_pred, 'print_result' : if_print_res, 'palette' : palette, 
        'batch_size' : batch_size}

    if call_args['no_save_vis'] and call_args['no_save_pred']:
        call_args['out_dir'] = ''

    if call_args['model'].endswith('.pth'):
        print_log('The model is a weight file, automatically '
                  'assign the model to --weights')
        call_args['weights'] = call_args['model']
        call_args['model'] = None

    init_kws = ['model', 'weights', 'device', 'palette']
    init_args = {}
    for init_kw in init_kws:
        init_args[init_kw] = call_args.pop(init_kw)

    return init_args, call_args

def output_demo_info_for_square_img(temp_timestamp, temp_dict):
    to_print = 'The detectoion result for %s is:\n'%temp_timestamp
    iter_keys = list(temp_dict.keys())
    iter_keys.sort()
    for key in iter_keys:
        temp_content = temp_dict.get(key)
        if not temp_content.get('pvals'):
            to_print += '%s has no boxes \n'%str(key)
        else:
            box_num = len(temp_content.get('pvals'))
            to_print += '%s has %d boxes: \n'%(str(key), box_num)
            boxes = temp_content.get('bboxes')
            pvals = temp_content.get('pvals')
            for i in range(box_num):
                temp_box = boxes[i]
                to_print += '[%.3f, %.3f, %.3f, %.3f] with p value of %.3f \n'%(temp_box[0], temp_box[1], temp_box[2], temp_box[3], pvals[i])
    print(to_print)
    
def output_demo_info_for_merge_img(temp_timestamp, temp_dict):
    to_print = 'The detectoion result for %s is:\n'%temp_timestamp
    if not temp_dict.get('pvals'):
        to_print += '%s has no boxes \n'%temp_timestamp
    else:
        box_num = len(temp_dict.get('pvals'))
        to_print += '%s has %d boxes: \n'%(temp_timestamp, box_num)
        boxes = temp_dict.get('bboxes')
        pvals = temp_dict.get('pvals')
        for i in range(box_num):
            temp_box = boxes[i]
            to_print += '[%.3f, %.3f, %.3f, %.3f] with p value of %.3f \n'%(temp_box[0], temp_box[1], temp_box[2], temp_box[3], pvals[i])
    print(to_print)

def image_demo_func(init_args, call_args):
    inferencer = DetInferencer(**init_args)
    inferencer(**call_args)

    if call_args['out_dir'] != '' and not (call_args['no_save_vis']
                                           and call_args['no_save_pred']):
        print_log(f'results have been saved at {call_args["out_dir"]}')

