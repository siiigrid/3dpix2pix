import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html
import numpy as np
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from util.animator3d import MedicalImageAnimator


#from calculate_metrics import Metrics

def generate(opt, epoch, how_many):
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.save_gif = False
    opt.which_epoch = epoch # to specify the model to use
    opt.how_many = how_many # how many images to generate


    data_loader = CreateDataLoader(opt) # I THINK IT IS FINE BECAUSE IT IS THE ONE WE WANTED (TRAIN)
    dataset = data_loader.load_data()
    model = create_model(opt)

    directory = "generated_for_eval"

    # Check if the directory exists, if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")
    else:
        print(f"Directory '{directory}' already exists.")



    subdirectory = opt.name[8:15]
    subdirectory_path = os.path.join(directory, subdirectory)

    if not os.path.exists(subdirectory_path):
        os.makedirs(subdirectory_path)
        print(f"Subdirectory '{subdirectory}' created inside '{directory}'.")
    else:
        print(f"Subdirectory '{subdirectory}' already exists inside '{directory}'.")
        

    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals() # OrderedDict([('real_A', real_A), ('fake_B', fake_B)]) and they are images in nympy

        img_path = model.get_image_paths()
        print('process image... %s' % img_path)
        gen_image = visuals['fake_B'] # (1, 32, 128, 128)





        # save image as npy
        if opt.which_direction == 'AtoB':
            if opt.dataset_name == "picai":
                npy_path =  data['B_paths'][0].split('/')[7].split('.')[0] + ".npy"
            elif opt.dataset_name == "prostatex":
                npy_path =  data['B_paths'][0].split('/')[6].split('.')[0] + ".npy"
        else:
            if opt.dataset_name == "picai":
                npy_path =  data['A_paths'][0].split('/')[7].split('.')[0] + ".npy"
            elif opt.dataset_name == "prostatex":
                npy_path =  data['A_paths'][0].split('/')[6].split('.')[0] + ".npy"
        final_npy_path = os.path.join(subdirectory_path, npy_path)
        print("image shape: ", gen_image.shape)
        np.save(final_npy_path, gen_image)

        


        # save image as gif
        if opt.save_gif:
            if opt.which_direction == 'AtoB':
                gif_path = data['B_paths'][0].split('/')[7].split('.')[0] + ".gif"
            else:
                gif_path = data['A_paths'][0].split('/')[7].split('.')[0] + ".gif"

            gen_image = np.squeeze(gen_image, axis=0)
            final_gif_path = os.path.join(subdirectory_path, gif_path)
            animator = MedicalImageAnimator(gen_image, [], 0, save=True, save_png = True)
            animate = animator.run(final_gif_path)

    return subdirectory_path

  