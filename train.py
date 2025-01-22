import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
import os
import wandb

print("Starting training...")


opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

config = {
    "momentum": opt.beta1,
    "optimizer": "Adam",
    "name": opt.name[8:15],
    "datast": opt.dataroot,  
    "attention_G": opt.attention_G,
}

run = wandb.init(project = "3dpix2pix", config = config)

model = create_model(opt)
visualizer = Visualizer(opt)
total_steps = 0

saving_path = os.path.join("./", opt.checkpoints_dir, opt.name, "out_metrics.txt")

with open(saving_path, "a") as outfile:
    outfile.write("-----Starting training-----\n")
for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    print("Epoch: ", epoch)
    epoch_start_time = time.time()
    epoch_iter = 0
    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(data)
        model.optimize_parameters()

        if total_steps % opt.display_freq == 0:
            visualizer.display_current_results(model.get_current_visuals(), epoch)

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                (epoch, total_steps))
            model.save('latest')

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
            (epoch, total_steps))
        model.save('latest')
        model.save(epoch)
    

    print('End of epoch %d / %d \t Time Taken: %d sec' %
        (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    if epoch > opt.niter:
        model.update_learning_rate()

    if epoch % opt.evaluate_network_freq == 0:
        stop_training, values_dict = model.evaluate_network(epoch)
        # Write evaluation results to the file
        wandb.log({"epoch": epoch, "psnr": values_dict["mean_psnr"], "ssim": values_dict["mean_ssim"], "nmse": values_dict["mean_nmse"]}, step = epoch)
        with open(saving_path, "a") as outfile:
            outfile.write(f"Evaluating epoch {epoch}:\n")
            outfile.write(f"{values_dict}\n")
        print(f"Evaluating epoch {epoch}:\n")
        print("Evaluation results: ", values_dict)
        if stop_training:
            outfile.write("-----Finished training-----\n")
            break