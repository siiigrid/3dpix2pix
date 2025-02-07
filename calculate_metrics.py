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

def scale_and_normalize(img, lower=0, upper=99.75, b_min=0, b_max=1):
    # Scale intensity range
    low_val = torch.quantile(img, lower / 100)
    high_val = torch.quantile(img, upper / 100)
    img = torch.clamp((img - low_val) / (high_val - low_val), 0, 1)  # Normalize to [0,1]
    img = img * (b_max - b_min) + b_min

    return img

def extract_patient_id(filepath):
    """Extracts the patient ID from the filename (assumes format: 'ProstateX-0062_*')"""
    return os.path.basename(filepath).split('_')[0]

picai_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda img: scale_and_normalize(img, 0, 99.75, 0, 1))
])

class Metrics:
    def __init__(self, real_dir, metrics, AtoB_name, dataset):
        self.real_dir = real_dir
        self.psnr = self.fid = self.ssim = self.nmse = False
        self.dataset_name = dataset
        self.image_class = AtoB_name[-3:]
        if self.dataset_name == "prostatex":
            if self.image_class == "h50": self.image_class = "b-value-50"
            if self.image_class == "400": self.image_class = "b-value-400"
            if self.image_class == "800": self.image_class = "b-value-800"
        
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
        self.data = []
        self.transform = picai_transforms
        if self.dataset_name == "picai":
            print("Using PICAI dataset")
            self.t2w_paths = sorted(glob(os.path.join(self.real_dir, '*t2w.nii.gz')))
            self.adc_paths = sorted(glob(os.path.join(self.real_dir, '*adc.nii.gz')))
            self.hbv_paths = sorted(glob(os.path.join(self.real_dir, '*hbv.nii.gz')))
            print(len(self.t2w_paths),len(self.adc_paths),len(self.hbv_paths), self.real_dir)
            
            
            # if len(self.t2w_paths) == 1400: self.seg_paths = sorted(glob(os.path.join('/mnt/gpid08/datasets/FLUTE/PICAI/mask_combined/binary_masks/training_set', '*.nii.gz')))
            # else:  self.seg_paths = sorted(glob(os.path.join('/mnt/gpid08/datasets/FLUTE/PICAI/mask_combined/binary_masks/test_set', '*.nii.gz')))

            for t2w, adc, hbv in zip(self.t2w_paths, self.adc_paths, self.hbv_paths):
                subject = os.path.basename(t2w)[:-11]
                assert subject == os.path.basename(adc)[:-11]
                assert subject == os.path.basename(hbv)[:-11]
                # assert subject == os.path.basename(seg)[:-7]
                self.data.append({"adc":adc, "t2w":t2w, "hbv": hbv,"subject_id": subject})


        elif self.dataset_name == "prostatex":
            print("Using ProstateX dataset")
            self.t2w_paths = sorted(glob(os.path.join(self.real_dir, '*t2_smoothed.nii.gz')))
            #self.adc_paths = sorted(glob(os.path.join(self.real_dir, '*adc_map_registered.nii.gz')))
            self.hbv_50_paths = sorted(glob(os.path.join(self.real_dir, '*-50.nii.gz')))
            self.hbv_400_paths = sorted(glob(os.path.join(self.real_dir, '*-400.nii.gz')))
            self.hbv_800_paths = sorted(glob(os.path.join(self.real_dir, '*-800.nii.gz')))
            self.all = sorted(glob(os.path.join(self.real_dir, '*')))
            print(len(self.t2w_paths), len(self.hbv_50_paths), len(self.hbv_400_paths), len(self.hbv_800_paths), self.real_dir)
            
            # Create sets of patient IDs for each modality
            t2w_ids = {extract_patient_id(p) for p in self.t2w_paths}
            #adc_ids = {extract_patient_id(p) for p in self.adc_paths}
            hbv_50_ids = {extract_patient_id(p) for p in self.hbv_50_paths}
            hbv_400_ids = {extract_patient_id(p) for p in self.hbv_400_paths}
            hbv_800_ids = {extract_patient_id(p) for p in self.hbv_800_paths}
        

            valid_patient_ids = t2w_ids  & hbv_50_ids & hbv_400_ids & hbv_800_ids

            # Filter file lists to keep only valid patients
            self.t2w_paths = [p for p in self.t2w_paths if extract_patient_id(p) in valid_patient_ids]
            # self.adc_paths = [p for p in self.adc_paths if extract_patient_id(p) in valid_patient_ids]
            self.hbv_50_paths = [p for p in self.hbv_50_paths if extract_patient_id(p) in valid_patient_ids]
            self.hbv_400_paths = [p for p in self.hbv_400_paths if extract_patient_id(p) in valid_patient_ids]
            self.hbv_800_paths = [p for p in self.hbv_800_paths if extract_patient_id(p) in valid_patient_ids]

            for t2w, hbv50, hbv400, hbv800 in zip(self.t2w_paths, self.hbv_50_paths, self.hbv_400_paths, self.hbv_800_paths):
                subject = os.path.basename(t2w)[:-19]
                #print(subject, os.path.basename(adc)[:-26], os.path.basename(hbv50)[:-18], os.path.basename(hbv400)[:-19], os.path.basename(hbv800)[:-19])
                #assert subject == os.path.basename(adc)[:-26]
                assert subject == os.path.basename(hbv50)[:-18]
                assert subject == os.path.basename(hbv400)[:-19]
                assert subject == os.path.basename(hbv800)[:-19]
                self.data.append({"t2w":t2w, "b-value-50": hbv50, "b-value-400": hbv400, "b-value-800": hbv800,"subject_id": subject})
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
            s_img2 = (s_img2 + 1) / 2 # normalize img2 to [0, 1]
            # print(s_img1.max(), s_img2.max(), s_img1.min(), s_img2.min())
            ssim_value, _ = ssim(s_img1, s_img2, full=True, data_range = 1)
            vals.append(ssim_value)
        return np.mean(vals)
  


    def calculate_nmse_def(self, img1, img2, dim = 3):
        mse = np.mean((img1 - img2) ** 2)
        norm = np.mean(img1 ** 2)
        # print(mse, norm, mse / norm)
        if norm == 0:
            print("Warning: norm is 0", mse)
            return "nan"
        nmse = mse / (norm)
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
                s_img2 = (s_img2 + 1) / 2 # normalize img2 to [0, 1]
                new_val = self.calculate_nmse_def(s_img1, s_img2, dim)
                if new_val != "nan":
                    vals.append(new_val)
            return np.mean(vals)
        return self.calculate_nmse_def(img1, img2, dim)
    

    def calculate_psnr_def(self, img1, img2, dim):
        mse = torch.mean((img1 - img2) ** 2).item()
        if mse == 0:
            return float('inf')  # Perfect match
        if dim == 2: max_pixel = 1.0
        else: max_pixel = torch.max(img1)
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        #print("psnr", max(img1), max(img2), min(img1), min(img2))
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
                s_img2 = (s_img2 + 1) / 2 # normalize img2 to [0, 1]
                #print(s_img1.max(), s_img2.max(), s_img1.min(), s_img2.min())
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
        print(nmse_values)      
        final_metrics = {"mean_psnr": np.mean(psnr_values), "mean_ssim": np.mean(ssim_values), "mean_nmse": np.mean(nmse_values)}
        return final_metrics

# Example usage
ground_truth_dir = '/mnt/work/datasets/FLUTE/PICAI/seq-256x256x32_test'
ground_truth_dir = '/mnt/work/datasets/FLUTE/prostatex_seq-256x256x32_test'
generated_dir = './generated_prostatex256'
metrics = ['ssim', 'psnr', "nmse"]
dataset_name = "prostatex"
#metrics = ['fid', 'pr']
AtoB_name = 't2w2h50'
m = Metrics(ground_truth_dir, metrics, AtoB_name, dataset = dataset_name)
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

