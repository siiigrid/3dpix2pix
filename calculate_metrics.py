import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from glob import glob
import torchvision.transforms as transforms
import nibabel as nib
import torch
from FID_implementation.fid_score import calculate_fid_given_paths
#from FID_implementation.precision_recall import compute_pr

picai_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        #transforms.ScaleIntensityRangePercentilesd(lower=0, upper=99.75, b_min=0, b_max=1)
    ]
)

class Metrics:
    def __init__(self, real_dir, metrics, AtoB_name):
        self.real_dir = real_dir
        self.psnr = self.fid = self.ssim = self.nmse = False
        self.image_class = AtoB_name[-3:]
        print(self.image_class)
        print(AtoB_name)
        self.AtoB_name = AtoB_name
        self.transform = picai_transforms
        self.data_paths_real = self.get_data_paths()
        for m in metrics:
            if m not in ['psnr', 'ssim', 'fid', 'nmse', 'precision', 'recall']:
                raise ValueError(f"Invalid metric: {m}")
            if m == 'fid':
                self.fid = True
                print("Calculating FID")
            if m == 'ssim':
                self.ssim = True
                print("Calculating SSIM")
            if m == 'nmse':
                self.nmse = True
                print("Calculating NMSE")
            if m == 'psnr':
                self.psnr = True
                print("Calculating PSNR")
            if m == 'pr':    
                self.pr = True
                print("Calculating Precision and Recall")


    def get_data_paths(self):
        self.t2w_paths = sorted(glob(os.path.join(self.real_dir, '*t2w.nii.gz')))
        self.adc_paths = sorted(glob(os.path.join(self.real_dir, '*adc.nii.gz')))
        self.hbv_paths = sorted(glob(os.path.join(self.real_dir, '*hbv.nii.gz')))

        self.data = []

        if len(self.t2w_paths) == 1400: self.seg_paths = sorted(glob(os.path.join('/mnt/gpid08/datasets/FLUTE/PICAI/mask_combined/binary_masks/training_set', '*.nii.gz')))
        else:  self.seg_paths = sorted(glob(os.path.join('/mnt/gpid08/datasets/FLUTE/PICAI/mask_combined/binary_masks/test_set', '*.nii.gz')))

        

        for t2w, adc, hbv, seg in zip(self.t2w_paths, self.adc_paths, self.hbv_paths, self.seg_paths):
            subject = os.path.basename(t2w)[:-11]
            assert subject == os.path.basename(adc)[:-11]
            assert subject == os.path.basename(hbv)[:-11]
            assert subject == os.path.basename(seg)[:-7]
            self.data.append({"t2w":t2w, "hbv":hbv, "adc": adc, "seg":seg, "subject_id": subject})

        return self.data
        
    
    def calculate_fid(self, path2, dim = 3):
        """
        Calculate FID (Frechet Inception Distance) between two sets of images.
        :param real_images: List of ground truth images
        :param generated_images: List of generated images
        :return: FID score
        """

        path2 = sorted(glob(os.path.join(path2, '*npy')))

        if self.image_class == "adc":
            return calculate_fid_given_paths([[self.adc_paths, "real"], [path2, "generated"]])
        if self.image_class == "t2w":
            return calculate_fid_given_paths([[self.t2w_paths, "real"], [path2, "generated"]])
        if self.image_class == "hbv":
            return calculate_fid_given_paths([[self.hbv_paths, "real"], [path2, "generated"]])
        return
    

    def calculate_ssim(self, img1, img2, dim = 3):
        """
        Calculate the SSIM (Structural Similarity Index) between two imagesm, for 2D or 3D images, in this case it is the same.
        :param img1: First image (ground truth)
        :param img2: Second image (generated image)
        :return: SSIM value
        """
        vals = []
        for i in zip(range(img1.shape[1])):
            s_img1 = img1[0, i, :, :]
            s_img2 = img2[0 ,i, :, :]
            s_img1 = s_img1.squeeze().cpu().numpy().astype(np.float32)
            s_img2 = s_img2.squeeze().cpu().numpy().astype(np.float32)
            #print(s_img1.max(), s_img2.max(), s_img1.min(), s_img2.min())
            ssim_value, _ = ssim(s_img1, s_img2, full=True, data_range = 2)
            vals.append(ssim_value)
        return np.mean(vals)
  


    def calculate_nmse_def(self, img1, img2, dim = 3):
        mse = np.mean((img1 - img2) ** 2)
        norm = np.mean(img1 ** 2)
        nmse = mse / norm
        return nmse
    

    def calculate_nmse(self, img1, img2, dim = 3):
        """
        Calculate the NMSE (Normalized Mean Squared Error) between two images, for 2D or 3D images.
        :param img1: First image (ground truth)
        :param img2: Second image (generated image)
        :return: NMSE value
        """
        if dim == 2:
            vals = []
            for i in zip(range(img1.shape[1])):
                s_img1 = img1[0, i, :, :]
                s_img2 = img2[0 ,i, :, :]
                s_img1 = s_img1.squeeze().cpu().numpy().astype(np.float32)
                s_img2 = s_img2.squeeze().cpu().numpy().astype(np.float32)
                vals.append(self.calculate_nmse_def(s_img1, s_img2, dim))
            return np.mean(vals)
        return self.calculate_nmse_def(img1, img2, dim)
    

    def calculate_psnr_def(self, img1, img2, dim):
        mse = torch.mean((img1 - img2) ** 2).item()
        if mse == 0:
            return float('inf')  # Perfect match
        if dim == 2: max_pixel = 255.0
        else: max_pixel = torch.max(img1)
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr

    def calculate_psnr(self, img1, img2, dim = 3):
        """
        Calculate the PSNR (Peak Signal-to-Noise Ratio) between two images, for 2D or 3D images.
        :param img1: First image (ground truth)
        :param img2: Second image (generated image)
        :return: PSNR value
        """
        if dim == 2:
            vals = []
            for i in zip(range(img1.shape[1])):
                s_img1 = img1[0, i, :, :]
                s_img2 = img2[0 ,i, :, :]
                vals.append(self.calculate_psnr_def(s_img1, s_img2, dim))
            return np.mean(vals)
        return self.calculate_psnr_def(img1, img2, dim)

    def calculate_pr(self, path2):
        if self.image_class == "adc":
            return compute_pr([[self.adc_paths, "real"], [path2, "generated"]])
        if self.image_class == "t2w":
            return compute_pr([[self.t2w_paths, "real"], [path2, "generated"]])
        if self.image_class == "hbv":
            return compute_pr([[self.hbv_paths, "real"], [path2, "generated"]])
        return

    def calculate_average_metrics(self, generated_dir, dim = 2):
        """
        Calculate the average specified metrics for a set of images.
        """
        self.generated_dir =  os.path.join(generated_dir, self.AtoB_name)
        psnr_values = []
        ssim_values = []
        nmse_values = []
        #if self.fid: final_fid = self.calculate_fid(self.generated_dir) 
        #if self.pr: precision, recall = self.calculate_pr(self.generated_dir)
        for data_dict in self.data_paths_real:
            #print(data_dict["subject_id"])
            generated_image_path = os.path.join(self.generated_dir, data_dict["subject_id"] + "_" + self.image_class + ".npy")
            #print(generated_image_path)
            if os.path.exists(generated_image_path):
                #print("Processing subject", data_dict["subject_id"])
                ground_truth = nib.load(data_dict[self.image_class]).get_fdata()
                ground_truth = self.transform(ground_truth).unsqueeze(0)
                generated_image = np.load(generated_image_path)
                generated_image = torch.from_numpy(generated_image)
                if self.psnr: psnr_values.append(self.calculate_psnr(ground_truth, generated_image, dim = 2))
                if self.ssim: ssim_values.append(self.calculate_ssim(ground_truth, generated_image, dim = 2))
                if self.nmse: nmse_values.append(self.calculate_nmse(ground_truth, generated_image, dim = 2)) 
                
        final_metrics = {"mean_psnr": np.mean(psnr_values), "mean_ssim": np.mean(ssim_values), "mean_nmse": np.mean(nmse_values)}
        return final_metrics

# Example usage
ground_truth_dir = '/mnt/work/datasets/FLUTE/PICAI/seq-128x128x32_test'
generated_dir = './generated_att'
metrics = ['ssim', 'psnr', "nmse"]
#metrics = ['fid', 'pr']
AtoB_name = 't2w2adc'
m = Metrics(ground_truth_dir, metrics, AtoB_name)
dims = [2]
for dim in dims:
    print("Metrics for", AtoB_name)
    print("Dimensions", dim)
    print(m.calculate_average_metrics(generated_dir=generated_dir, dim = dim))


# If your images are normalized (e.g., pixel values are in [0, 1]), scale them back to [0, 255] before computing PSNR.
# reads real images from folder and transforms to according shape [channel, z, x, y]
# generated images have to be in npy format and have the same shape as the real images



# TO DO:
# - METRICS CLAUDIA TOLD ME

