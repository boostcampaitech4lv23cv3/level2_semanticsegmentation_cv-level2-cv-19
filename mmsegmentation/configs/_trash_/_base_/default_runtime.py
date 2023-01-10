# from datetime import datetime, timedelta

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook'),
        # dict(type='PaviLoggerHook') # for internal services

        # WandB Logging
        dict(type='WandbLoggerHook',
             init_kwargs=dict(
                 project='semantic-segmentation',
                 entity='cv_19',
                 # name=f'exp_{datetime.strftime(datetime.now() + timedelta(hours=9), "%Y%m%d%H%M%S")}'
                 name="exp_202301021544"
             ))
    ])
# yapf:enable
# custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True