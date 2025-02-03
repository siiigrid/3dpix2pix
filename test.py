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

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test

# PRINT NAME OF MODEL TO SEE IF IT IS TEST_MODEL OR 3D_PIX2PIX WHEN DOING TESTING
# IN THE G, PUT IF TEST: RETURN ENCODED IMG


def plot_3d_tensor(tensor):
    """
    Plots a 3D tensor (depth, height, width) using an interactive 3D plot.

    Parameters:
        tensor (numpy.ndarray): A 3D numpy array of shape (depth, height, width).
    """
    if len(tensor.shape) != 3:
        raise ValueError("Input tensor must be a 3D array.")

    # Get the indices of the non-zero elements
    depth, height, width = tensor.shape

    # Get the values of the non-zero elements
    values = tensor #[ tensor != 0]

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    width_indices, height_indices, depth_indices = np.meshgrid(np.arange(width), np.arange(height), np.arange(depth), indexing='ij')

    scatter = ax.scatter(width_indices, height_indices, depth_indices, c=values, cmap='viridis', marker='o')

    # Add color bar to indicate the values
    color_bar = plt.colorbar(scatter, ax=ax, pad=0.1)
    color_bar.set_label('Values')

    # Label axes
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    ax.set_zlabel('Depth')

    # Set title
    ax.set_title('3D Tensor Visualization')
    plt.savefig('patata.png')
    # Show the plot
    plt.show()






directory = "generated" + opt.name[15:]

# Check if the directory exists, if not, create it
if not os.path.exists(directory):
    os.makedirs(directory)
    print(f"Directory '{directory}' created.")
else:
    print(f"Directory '{directory}' already exists.")


base_directory = directory
subdirectory = opt.name[8:15]
subdirectory_path = os.path.join(base_directory, subdirectory)

if not os.path.exists(subdirectory_path):
    os.makedirs(subdirectory_path)
    print(f"Subdirectory '{subdirectory}' created inside '{base_directory}'.")
else:
    print(f"Subdirectory '{subdirectory}' already exists inside '{base_directory}'.")

    

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
        npy_path =  data['B_paths'][0].split('/')[7].split('.')[0] + ".npy"
    else:
        npy_path =  data['A_paths'][0].split('/')[7].split('.')[0] + ".npy"
    final_npy_path = os.path.join(subdirectory_path, npy_path)
    np.save(final_npy_path, gen_image)

    


    # save image as gif
    if opt.save_gif == True:
        if opt.which_direction == 'AtoB':
            gif_path = data['B_paths'][0].split('/')[7].split('.')[0] + ".gif"
        else:
            gif_path = data['A_paths'][0].split('/')[7].split('.')[0] + ".gif"

        gen_image = np.squeeze(gen_image, axis=0)
        final_gif_path = os.path.join(subdirectory_path, gif_path)
        animator = MedicalImageAnimator(gen_image, [], 0, save=True, save_png = True)
        animate = animator.run(final_gif_path)

    
    #visualizer.save_images(webpage, visuals, img_path) 
webpage.save()
"""   image_numpy = data['A'].cpu().squeeze().numpy()

    animator = MedicalImageAnimator(image_numpy, [], 0, save=True, save_png = True)
    animate = animator.run(f'./prova.gif')
"""