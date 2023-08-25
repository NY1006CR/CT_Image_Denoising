import torch.utils.data.distributed
import torch.utils.data.distributed
from noise2sim.modeling.architectures import build_architecture

def savemodel(model, model_weight, save_path):
    # 将model_weight 转存为Libtorch格式
    model_weight_save = model_weight
    checkpoint = torch.load(model_weight_save)
    state_dict = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # Initialize the feature module with encoder_q of moco.
        if k.startswith('module'):
            # remove prefix
            state_dict[k[len('module.'):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
    model.load_state_dict(state_dict)
    model.eval()

    # Libtorch保存
    input = torch.ones(1, 1, 800, 800)
    trace_model = torch.jit.trace(model, input)
    # 保存路径
    trace_model.save(save_path)

# 定义模型
model = dict(
    type="common_denoiser",
    base_net=dict(
        type="unet2",
        n_channels=1,
        n_classes=1,
        activation_type="relu",
        bilinear=False,
        residual=True,
        use_bn=True
    ),
    denoiser_head=dict(
        loss_type="l2",
        loss_weight={"l2": 1},
    ),
    weight=None,
)

# 定义权重
model_weight = 'script_model/checkpoint_final.pth.tar'
output_dir="./output/ns8"

# create model
model = build_architecture(model)

savemodel(model, model_weight, output_dir)


