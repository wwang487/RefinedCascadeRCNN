import os
import os.path as osp
import warnings
from copy import deepcopy

from mmengine import ConfigDict
from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mmdet.engine.hooks.utils import trigger_visualization_hook, trigger_visualization_hook_w_dict
from mmdet.evaluation import DumpDetResults
from mmdet.registry import RUNNERS
from mmdet.utils import setup_cache_size_limit_of_dynamo

def create_testing_parse(config_file_path, checkpoint_file_path, eval_save_folder, \
                        pred_save_file_path, vis_save_folder, launcher = 'none', \
                        local_rank = 0, is_tta = False, is_show = True):
    if eval_save_folder:
        os.makedirs(eval_save_folder, exist_ok = True)
    args_dict = {'config': config_file_path,
                 'checkpoint': checkpoint_file_path,
                 'work_dir': eval_save_folder,
                 'out': pred_save_file_path, 
                 'show': is_show,
                 'show_dir': vis_save_folder,
                 'launcher': launcher, 'tta': is_tta,
                 'local_rank': local_rank}
    
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args_dict.get("local_rank"))
    if 'RANK' not in os.environ:
        os.environ['RANK'] = str(0)

    return args_dict

def test_func(args_dict):
    # Reduce the number of repeated compilations and improve
    # testing speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args_dict.get('config'))
    cfg.launcher = args_dict.get('launcher')
    
    # if args.cfg_options is not None:
    #    cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args_dict.get("work_dir") is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args_dict.get("work_dir")
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args_dict.get('config')))[0])

    cfg.load_from = args_dict.get("checkpoint")

    if args_dict.get("show") or args_dict.get("show_dir"):
        cfg = trigger_visualization_hook_w_dict(cfg, args_dict)

    if args_dict.get("tta"):

        if 'tta_model' not in cfg:
            warnings.warn('Cannot find ``tta_model`` in config, '
                          'we will set it as default.')
            cfg.tta_model = dict(
                type='DetTTAModel',
                tta_cfg=dict(
                    nms=dict(type='nms', iou_threshold=0.5), max_per_img=100))
        if 'tta_pipeline' not in cfg:
            warnings.warn('Cannot find ``tta_pipeline`` in config, '
                          'we will set it as default.')
            test_data_cfg = cfg.test_dataloader.dataset
            while 'dataset' in test_data_cfg:
                test_data_cfg = test_data_cfg['dataset']
            cfg.tta_pipeline = deepcopy(test_data_cfg.pipeline)
            flip_tta = dict(
                type='TestTimeAug',
                transforms=[
                    [
                        dict(type='RandomFlip', prob=1.),
                        dict(type='RandomFlip', prob=0.)
                    ],
                    [
                        dict(
                            type='PackDetInputs',
                            meta_keys=('img_id', 'img_path', 'ori_shape',
                                       'img_shape', 'scale_factor', 'flip',
                                       'flip_direction'))
                    ],
                ])
            cfg.tta_pipeline[-1] = flip_tta
        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # add `DumpResults` dummy metric
    if args_dict.get("out") is not None:
        assert args_dict.get("out").endswith(('.pkl', '.pickle')), \
            'The dump file must be a pkl file.'
        runner.test_evaluator.metrics.append(
            DumpDetResults(out_file_path=args_dict.get("out")))

    # start testing
    runner.test()
    
def get_test_img_list(test_file):
    with open(test_file, 'r') as file:
        # Read lines from the file
        lines = file.readlines()

    # Strip newline characters from each line
    lines = [line.strip() for line in lines]
    return lines

def compute_prec_rec(box_dict):
    sum_TT, sum_TF, sum_FT = 0, 0, 0
    for k in box_dict.keys():
        temp_content = box_dict.get(k)
        for t in temp_content.keys():
            sub_content = temp_content.get(t)
            sum_TT += sub_content.get('TT')
            sum_TF += sub_content.get('TF')
            sum_FT += sub_content.get('FT')
    return sum_TT, sum_TF, sum_FT
