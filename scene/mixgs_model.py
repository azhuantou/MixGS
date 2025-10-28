import torch
import torch.nn as nn
import torch.nn.functional as F
from scene.network import GSDecoder, GSEncoder
import os
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func


class MixGSModel:
    def __init__(
            self, 
            hash_args,
            net_args,
    ):
        self.encoder = GSEncoder(**hash_args).cuda()
        self.spatial_dim = self.encoder.canonical_level_dim * self.encoder.canonical_num_levels
        self.mlp_dim = 10
        self.decoder = GSDecoder(spatial_in_dim=self.spatial_dim, mlp_in_dim=self.mlp_dim, **net_args).cuda()

        self.decoder_lr_scale = 50.0
        self.encoder_lr_scale = 100.0

    def step(self, data, pose):
        coords = data[0]
        scale_input = data[1]
        rotate_input = data[2]

        spatial_h, temporal_h = self.encoder(coords, pose)
        color, rotation, scaling, opacity = self.decoder(spatial_h, temporal_h, scale_input, rotate_input)
            
        return {
            "d_color": color,
            "d_rotation": rotation, 
            "d_scaling": scaling,
            "d_opacity": opacity,
        }
    
    def train_setting(self, training_args):
        self.decoder_lr_scale = training_args.decoder_lr_scale
        self.encoder_lr_scale = training_args.encoder_lr_scale

        l = [
            {'params': list(self.decoder.parameters()),
             'lr': training_args.position_lr_init * self.decoder_lr_scale,
             "name": "decoder"},
            {'params': list(self.encoder.parameters()),
             'lr': training_args.position_lr_init * self.encoder_lr_scale,
             "name": "encoder"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.decoder_lr_scheduler = get_expon_lr_func(lr_init=training_args.position_lr_init * self.decoder_lr_scale,
                                                       lr_final=training_args.position_lr_final,
                                                       lr_delay_mult=training_args.position_lr_delay_mult,
                                                       max_steps=training_args.decoder_lr_max_steps)
        self.encoder_lr_scheduler = get_expon_lr_func(lr_init=training_args.position_lr_init * self.encoder_lr_scale,
                                                       lr_final=training_args.position_lr_final * self.encoder_lr_scale,
                                                       lr_delay_mult=training_args.position_lr_delay_mult,
                                                       max_steps=training_args.decoder_lr_max_steps)

    def save_weights(self, model_path, iteration, is_best=False):
        if is_best:
            out_weights_path = os.path.join(model_path, "decoder/iteration_best")
            os.makedirs(out_weights_path, exist_ok=True)
            with open(os.path.join(out_weights_path, "iter.txt"), "w") as f:
                f.write("Best iter: {}".format(iteration))
        else:
            out_weights_path = os.path.join(model_path, "decoder/iteration_{}".format(iteration))
            os.makedirs(out_weights_path, exist_ok=True)
        torch.save((self.encoder.state_dict(), self.decoder.state_dict()), os.path.join(out_weights_path, 'decoder.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "decoder"))
            weights_path = os.path.join(model_path, "decoder/iteration_{}/decoder.pth".format(loaded_iter))
        else:
            loaded_iter = iteration
            weights_path = os.path.join(model_path, "decoder/iteration_{}/decoder.pth".format(loaded_iter))

        print("Load weight:", weights_path)
        grid_weight, network_weight = torch.load(weights_path, map_location='cuda')
        self.decoder.load_state_dict(network_weight)
        self.encoder.load_state_dict(grid_weight)

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "decoder":
                lr = self.decoder_lr_scheduler(iteration)
                param_group['lr'] = lr
            elif param_group['name'] == "encoder":
                lr = self.encoder_lr_scheduler(iteration)
                param_group['lr'] = lr

