import os

import logging
import os.path as osp

from mmdet.registry import RUNNERS
from mmengine.runner import Runner
from mmdet.utils import setup_cache_size_limit_of_dynamo


from mmengine.config import Config, DictAction
from mmengine.logging import print_log

def create_training_parse(train_config_file, model_log_save_file, is_amp, 
                          is_auto_scale_lr, resume = None, launcher = 'none',
                          local_rank = 0):
    args_dict = {'config':train_config_file, 'work_dir': model_log_save_file, 
                 'is_auto_scale_lr' : is_auto_scale_lr, 'resume': resume,
                 'amp': is_amp, 'launcher': launcher, 'local_rank': local_rank}
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args_dict.get("local_rank"))
    if 'RANK' not in os.environ:
        os.environ['RANK'] = str(0)
    return args_dict

def train_func(args_dict):
    # Reduce the number of repeated compilations and improve
    # training speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args_dict.get("config"))
    cfg.launcher = args_dict.get("launcher")
    # work_dir is determined in this priority: CLI > segment in file > filename
    if args_dict.get("work_dir") is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args_dict.get("work_dir")
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args_dict.get("config"))[0]))

    # enable automatic-mixed-precision training
    if args_dict.get("amp") is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # enable automatically scaling LR
    if args_dict.get("auto_scale_lr"):
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    # resume is determined in this priority: resume from > auto_resume
    if args_dict.get("resume") == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args_dict.get("resume") is not None:
        cfg.resume = True
        cfg.load_from = args_dict.get("resume")

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()

def modify_runtime_file(output_folder, output_file, scope = 'mmdet', timer = 'IterTimerHook', logger_type = 'LoggerHook',
                       logger_interval = 200, param_scheduler = 'ParamSchedulerHook',
                       checkpoint_type = 'CheckpointHook',
                       checkpoint_interval = 10, sampler_seed = 'DistSamplerSeedHook',
                       visulization = 'DetVisualizationHook', cudnn_benchmark = False,
                       mp_cfg_start_methods = 'fork', mp_cfg_num_threads = 0,
                       dist_cfg = 'nccl', vis_backends = 'LocalVisBackend', 
                       visualizer = 'DetLocalVisualizer', log_processor_type = 'LogProcessor',
                       log_processor_window = 50, log_processor_by_epoch = True,
                       log_level = 'INFO', load_from = None, resume = False):
    file_content = ''
    file_content += "default_scope = '%s'\n"%scope
    file_content += '\n'
    file_content += "default_hooks = dict(\n"
    file_content += "\ttimer=dict(type='%s'),\n"%timer
    file_content += "\tlogger=dict(type='%s', interval=%d),\n"%(logger_type, logger_interval)
    file_content += "\tparam_scheduler=dict(type='%s'),\n" % param_scheduler
    file_content += "\tcheckpoint=dict(type='%s', interval=%d),\n"%(checkpoint_type, checkpoint_interval)
    file_content += "\tsampler_seed=dict(type='%s'),\n"%sampler_seed
    file_content += "\tvisualization=dict(type='%s'))\n"%visulization
    file_content += '\n'
    file_content += "env_cfg = dict(\n"
    file_content += "\tcudnn_benchmark=%s,\n"%(str(cudnn_benchmark))
    file_content += "\tmp_cfg=dict(mp_start_method='%s', opencv_num_threads=%d),\n"%(mp_cfg_start_methods, mp_cfg_num_threads)
    file_content += "\tdist_cfg=dict(backend='%s'),\n"%dist_cfg
    file_content += ")\n"
    file_content += "vis_backends = [dict(type='%s')]\n"%vis_backends
    file_content += "visualizer = dict(\n"
    file_content += "\ttype='%s', vis_backends=vis_backends, name='visualizer')\n"%visualizer
    file_content += "log_processor = dict(type='%s', window_size=%d, by_epoch=%s)\n"%(log_processor_type,\
                                                          log_processor_window,str(log_processor_by_epoch))
    file_content += "\n"
    file_content += "log_level = '%s'\n"%log_level
    file_content += "load_from = %s\n"%str(load_from)
    file_content += "resume = %s\n"%str(resume)
    file_content += "\n"
    
    os.makedirs(output_folder, exist_ok=True)
    with open(output_folder + output_file, 'w') as f:
        print(file_content, file=f)
    f.close()

def modify_schedule_file(output_folder, output_file,
                         train_cfg_type = 'EpochBasedTrainLoop', max_epochs = 100, val_interval = 1,
                         val_cfg_type = 'ValLoop', test_cfg_type = 'TestLoop',
                         linearLRParams = [0.001, False, 0, 500],
                         multiStepLRParams = [0, 12, True, 8, 11, 0.10],
                         optimizerParams = ['SGD', 0.02, 0.9, 0.0001], 
                         autoScaleParams = [False, 16]):
    schedule_contents = ""
    schedule_contents += "train_cfg = dict(type='%s', max_epochs=%d, val_interval=%d)\n"%(train_cfg_type, max_epochs, val_interval)
    schedule_contents += "val_cfg = dict(type='%s')\n"%val_cfg_type
    schedule_contents += "test_cfg = dict(type='%s')\n"%test_cfg_type
    schedule_contents += "param_scheduler = [\n"
    schedule_contents += "\tdict(\n"
    schedule_contents += "\t\ttype='LinearLR', start_factor=%f, by_epoch=%s, begin=%d, end=%d),\n"%(
                         linearLRParams[0], linearLRParams[1], linearLRParams[2], linearLRParams[3]
                         )
    schedule_contents += "\tdict(\n"
    schedule_contents += "\t\ttype='MultiStepLR',\n"
    schedule_contents += "\t\tbegin=%d,\n"%multiStepLRParams[0]
    schedule_contents += "\t\tend=%d,\n"%multiStepLRParams[1]
    schedule_contents += "\t\tby_epoch=%s,\n"%str(multiStepLRParams[2])
    schedule_contents += "\t\tmilestones=[%d,%d],\n"%(multiStepLRParams[3],multiStepLRParams[4])
    schedule_contents += "\t\tgamma=%f)\n"%multiStepLRParams[5]
    schedule_contents += "]\n"
    schedule_contents += "\n"
    
    schedule_contents += "optim_wrapper = dict(\n"
    schedule_contents += "\ttype='OptimWrapper',\n"
    schedule_contents += "\toptimizer=dict(type='%s', lr=%f, momentum=%f, weight_decay=%f))\n"%(
                         optimizerParams[0], optimizerParams[1], optimizerParams[2], optimizerParams[3]
                         )
    schedule_contents += "\n"
    schedule_contents += "auto_scale_lr = dict(enable=%s, base_batch_size=%d)\n"%(str(autoScaleParams[0]), autoScaleParams[1])
    os.makedirs(output_folder, exist_ok=True)
    with open(output_folder + output_file, 'w') as f:
        print(schedule_contents, file=f)
    f.close()

def modify_config_file(output_folder, output_file,
                       model_file =  '../_base_/models/cascade_rcnn_r50_fpn_voc.py',
                       dataset_file = '../_base_/datasets/voc0712.py', 
                       schedule_file = '../_base_/schedules/schedule_1x.py', 
                       runtime_file = '../_base_/default_runtime.py',
                       runner_type = 'EpochBasedRunner', max_epochs = 50):
    config_content = ''
    config_content += "_base_ = [\n"
    config_content += "\t'%s',\n"%model_file
    config_content += "\t'%s',\n"%dataset_file
    config_content += "\t'%s',\n"%schedule_file
    config_content += "\t'%s',\n"%runtime_file
    config_content += "]\n"
    config_content += "\n"
    config_content += "runner = dict(type='%s', max_epochs=%d)\n"%(runner_type, max_epochs)
    os.makedirs(output_folder, exist_ok=True)
    with open(output_folder + output_file, 'w') as f:
        print(config_content, file=f)
    f.close()