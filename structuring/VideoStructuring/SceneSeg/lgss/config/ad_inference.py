experiment_name = "ad300"
experiment_description = "scene segmentation with all modality"
# overall confg
data_root = '../data/ad300'
shot_frm_path = data_root + "/shot_txt"
shot_num = 4  # even
seq_len = 2  # even
gpus = "0,1,2,3,4,5,6,7"

# dataset settings
dataset = dict(
    name="all",
    mode=['place', 'aud'],
)
# model settings
model = dict(
    name='LGSS',
    sim_channel=512,  # dim of similarity vector
    place_feat_dim=2048,
    cast_feat_dim=512,
    act_feat_dim=512,
    aud_feat_dim=512,
    aud=dict(cos_channel=512),
    bidirectional=True,
    lstm_hidden_size=512,
    ratio=[0.8, 0, 0, 0.2]
    )

# optimizer
optim = dict(name='Adam',
             setting=dict(lr=1e-2, weight_decay=5e-4))
stepper = dict(name='MultiStepLR',
               setting=dict(milestones=[15]))
loss = dict(weight=[0.5, 5])

# runtime settings
resume = None
trainFlag = 0
testFlag = 1
batch_size = 16
epochs = 30
logger = dict(log_interval=200, logs_dir="../run/{}".format(experiment_name))
data_loader_kwargs = dict(num_workers=1, pin_memory=True, drop_last=True)
