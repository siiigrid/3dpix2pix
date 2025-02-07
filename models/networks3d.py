import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np
import math
###############################################################################
# Functions
###############################################################################


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm3d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm3d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm3d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, gpu_ids=[], attention_G='normal'):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 5, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids, attention_G=attention_G)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids, attention_G = attention_G)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) > 0:
        netG.cuda(device=gpu_ids[0])
    netG.apply(weights_init)
    return netG


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    if use_gpu:
        netD.cuda(device=gpu_ids[0])
    netD.apply(weights_init)
    return netD


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        self.device = torch.device('cuda:0')
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input.to(self.device), target_tensor.to(self.device))

    
# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm3d, use_dropout=False, gpu_ids=[], attention_G = 'normal'):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        # currently support only input_nc == output_nc
        assert(input_nc == output_nc)

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, norm_layer=norm_layer, innermost=True, attention_G=attention_G)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer, use_dropout=use_dropout, attention_G=attention_G)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer, attention_G=attention_G)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer, attention_G=attention_G)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, unet_block, norm_layer=norm_layer, attention_G=attention_G)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer, attention_G=attention_G)

        self.model = unet_block

    def forward(self, input):
        #print(f"Input shape to UnetGenerator: {input.shape}")  # Log the initial input shape
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            #output = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
            output = self.model(input)
        else:
            output = self.model(input)
        #print(f"Output shape from UnetGenerator: {output.shape}")  # Log the final output shape
        return output



# Rotary Positional Embedding Function
def rotary_embedding(pos, dim):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    inv_freq = inv_freq.to('cuda')
    pos = pos.to('cuda')
    sinusoid_inp = torch.einsum("i,j->ij", pos, inv_freq)  # Outer product
    emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
    return emb.T # i transpose it odtherwise i have problems with the shape





class RoPEAttention3D(nn.Module):
    def __init__(self, in_channels, num_heads=4):
        super(RoPEAttention3D, self).__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"

        self.query = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.key = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.value = nn.Conv3d(in_channels, in_channels, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)
        self.out = nn.Conv3d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        B, C, D, H, W = x.size()
        #print("x.size()", x.size())
        assert C % self.num_heads == 0, "Channels must be divisible by num_heads"

        # Generate query, key, value
        q = self.query(x).view(B, self.num_heads, self.head_dim, -1)  # B x H x D/H x N
        k = self.key(x).view(B, self.num_heads, self.head_dim, -1)
        v = self.value(x).view(B, self.num_heads, self.head_dim, -1)

        # Generate positional embeddings
        pos = torch.linspace(-1, 1, D * H * W, device=x.device)
        #print("pos.size()", pos.size())
        #print("q.size()", q.size())
        pos_emb = rotary_embedding(pos, self.head_dim)
        #print("pos_emb.size()", pos_emb.size())

        # Apply RoPE
        q = q * pos_emb.unsqueeze(0).unsqueeze(0).to(q.device)  # Apply rotation to query
        k = k.to(q.device) * pos_emb.unsqueeze(0).unsqueeze(0).to(q.device)  # Apply rotation to key

        # Attention computation
        attn = torch.einsum("bhcn,bhcm->bhnm", q, k)  # Dot product
        attn = self.softmax(attn)

        out = torch.einsum("bhnm,bhcm->bhcn", attn, v.to(q.device))  # Weighted sum
        #print("out.size()", out.size())
        out = out.reshape(B, C, D, H, W)  # Reshape back
        return self.out(out+ x)  # Residual connection



class RoPEMultiHeadCrossAttention3D(nn.Module):
    def __init__(self, in_channels, num_heads=4):
        super(RoPEMultiHeadCrossAttention3D, self).__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"

        # Linear projections for query, key, value
        self.query = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.key = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.value = nn.Conv3d(in_channels, in_channels, kernel_size=1)

        # Output projection
        self.out = nn.Conv3d(in_channels, in_channels*2, kernel_size=1)

        # Softmax for attention weights
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # Deconcatenate input tensor into query and key-value features
        query_features, key_value_features = torch.chunk(x, 2, dim=1)
        # print("query_features.size()", query_features.size())
        # print("key_value_features.size()", key_value_features.size())	


        B, C, D, H, W = query_features.size()
        assert C % self.num_heads == 0, "Channels must be divisible by num_heads"

        # Generate query, key, value
        ("query_features.size()", query_features.size())
        # print(self.query)
        q = self.query(query_features).view(B, self.num_heads, self.head_dim, -1)
        k = self.key(key_value_features).view(B, self.num_heads, self.head_dim, -1)
        v = self.value(key_value_features).view(B, self.num_heads, self.head_dim, -1)
        # print("i arrived after computing q, k, v")

        # Generate positional embeddings
        pos = torch.linspace(-1, 1, D * H * W, device=x.device)
        pos_emb = rotary_embedding(pos, self.head_dim)

        # print("positional embedings computed")

        # Apply RoPE
        q = q * pos_emb.unsqueeze(0).unsqueeze(0)
        k = k * pos_emb.unsqueeze(0).unsqueeze(0)

        # print("rope applied")

        # Attention computation
        attn = torch.einsum("bhcn,bhcm->bhnm", q, k)
        attn = self.softmax(attn)

        # print("attention computed")

        out = torch.einsum("bhnm,bhcm->bhcn", attn, v)
        out = out.reshape(B, C, D, H, W)
        # print("out.size()", out.size())
        # print("query_features.size()", query_features.size())
        # print((out + query_features).size())

        output = self.out(out + query_features)

        return output # Residual connection


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm3d, use_dropout=False, attention_G='normal'):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.attention_G = attention_G
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        downconv = nn.Conv3d(outer_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose3d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if attention_G == "self_begining":
                down = [RoPEAttention3D(outer_nc, num_heads=1)] + down

            
            # Use attention_G in skip connections
            if attention_G == "self":
                up = [RoPEAttention3D(inner_nc*2)] + up
            elif attention_G == "cross":
                up = [RoPEMultiHeadCrossAttention3D(inner_nc)] + up # here not *2 because we are not concatenating the features with the skip connection

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        #print(f"Input shape at this block: {x.shape}")  # Log the input shape
        if self.outermost:
            output = self.model(x)
            #print(f"Output shape at outermost block: {output.shape}")  # Log the output shape
            return output
        else:
            output = self.model(x)
            #print(f"Output shape before concatenation: {output.shape}")  # Log the intermediate output shape
            concatenated = torch.cat([output, x], 1)
            #print(f"Output shape after concatenation: {concatenated.shape}")  # Log the concatenated shape
            return concatenated

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d 
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequence = [
            nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), 
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            #return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
            return self.model(input)
        else:
            return self.model(input)