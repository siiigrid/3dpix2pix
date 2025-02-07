import os.path
import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
from glob import glob
import nibabel as nib
import torch

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass


def scale_and_normalize(img, lower=0, upper=99.75, b_min=0, b_max=1):
    # Scale intensity range
    low_val = torch.quantile(img, lower / 100)
    high_val = torch.quantile(img, upper / 100)
    img = torch.clamp((img - low_val) / (high_val - low_val), 0, 1)  # Normalize to [0,1]
    img = img * (b_max - b_min) + b_min

    return img


def scale_transform(img):
    return scale_and_normalize(img, 0, 99.75, 0, 1)  # Use a named function

picai_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(scale_transform)  #  Named function instead of lambda
])

def extract_patient_id(filepath):
    """Extracts the patient ID from the filename (assumes format: 'ProstateX-0062_*')"""
    return os.path.basename(filepath).split('_')[0]

class ProstateDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.device = torch.device('cuda:0')

        self.data = []
        self.transform = picai_transforms

        if self.opt.dataset_name == "picai":
            print("Using PICAI dataset")
            self.t2w_paths = sorted(glob(os.path.join(self.root, '*t2w.nii.gz')))
            self.adc_paths = sorted(glob(os.path.join(self.root, '*adc.nii.gz')))
            self.hbv_paths = sorted(glob(os.path.join(self.root, '*hbv.nii.gz')))
            print(len(self.t2w_paths),len(self.adc_paths),len(self.hbv_paths), self.root)
            
            
            # if len(self.t2w_paths) == 1400: self.seg_paths = sorted(glob(os.path.join('/mnt/gpid08/datasets/FLUTE/PICAI/mask_combined/binary_masks/training_set', '*.nii.gz')))
            # else:  self.seg_paths = sorted(glob(os.path.join('/mnt/gpid08/datasets/FLUTE/PICAI/mask_combined/binary_masks/test_set', '*.nii.gz')))

            for t2w, adc, hbv in zip(self.t2w_paths, self.adc_paths, self.hbv_paths):
                subject = os.path.basename(t2w)[:-11]
                assert subject == os.path.basename(adc)[:-11]
                assert subject == os.path.basename(hbv)[:-11]
                # assert subject == os.path.basename(seg)[:-7]
                self.data.append({"adc":adc, "t2w":t2w, "hbv": hbv,"subject_id": subject})
            print("len de data", len(self.data))


        elif self.opt.dataset_name == "prostatex":
            print("Using ProstateX dataset")
            self.t2w_paths = sorted(glob(os.path.join(self.root, '*t2_smoothed.nii.gz')))
            #self.adc_paths = sorted(glob(os.path.join(self.root, '*adc_map_registered.nii.gz')))
            self.hbv_50_paths = sorted(glob(os.path.join(self.root, '*-50.nii.gz')))
            self.hbv_400_paths = sorted(glob(os.path.join(self.root, '*-400.nii.gz')))
            self.hbv_800_paths = sorted(glob(os.path.join(self.root, '*-800.nii.gz')))
            self.all = sorted(glob(os.path.join(self.root, '*')))
            print(len(self.t2w_paths),len(self.hbv_50_paths), len(self.hbv_400_paths), len(self.hbv_800_paths), self.root)
            
            # Create sets of patient IDs for each modality
            t2w_ids = {extract_patient_id(p) for p in self.t2w_paths}
            #adc_ids = {extract_patient_id(p) for p in self.adc_paths}
            hbv_50_ids = {extract_patient_id(p) for p in self.hbv_50_paths}
            hbv_400_ids = {extract_patient_id(p) for p in self.hbv_400_paths}
            hbv_800_ids = {extract_patient_id(p) for p in self.hbv_800_paths}

            valid_patient_ids = t2w_ids & hbv_50_ids & hbv_400_ids & hbv_800_ids

            # Filter file lists to keep only valid patients
            self.t2w_paths = [p for p in self.t2w_paths if extract_patient_id(p) in valid_patient_ids]
            #self.adc_paths = [p for p in self.adc_paths if extract_patient_id(p) in valid_patient_ids]
            self.hbv_50_paths = [p for p in self.hbv_50_paths if extract_patient_id(p) in valid_patient_ids]
            self.hbv_400_paths = [p for p in self.hbv_400_paths if extract_patient_id(p) in valid_patient_ids]
            self.hbv_800_paths = [p for p in self.hbv_800_paths if extract_patient_id(p) in valid_patient_ids]

            for t2w, hbv50, hbv400, hbv800 in zip(self.t2w_paths, self.hbv_50_paths, self.hbv_400_paths, self.hbv_800_paths):
                subject = os.path.basename(t2w)[:-19]
                #print(subject, os.path.basename(adc)[:-26], os.path.basename(hbv50)[:-18], os.path.basename(hbv400)[:-19], os.path.basename(hbv800)[:-19])
                # assert subject == os.path.basename(adc)[:-26]
                assert subject == os.path.basename(hbv50)[:-18]
                assert subject == os.path.basename(hbv400)[:-19]
                assert subject == os.path.basename(hbv800)[:-19]
                self.data.append({"t2w":t2w, "h50": hbv50, "400": hbv400, "800": hbv800,"subject_id": subject})
            




    def __getitem__(self, index):
        #returns samples of dimension [channels, z, x, y]
        sample = self.data[index]
        nameA = self.opt.name[8:11]
        nameB = self.opt.name[12:15]


        nifti_data_A = nib.load(sample[nameA])
        A = nifti_data_A.get_fdata()
        nifti_data_B = nib.load(sample[nameB])
        B = nifti_data_B.get_fdata()

        A = self.transform(A).unsqueeze(0)
        B = self.transform(B).unsqueeze(0)


        return {
                'A' : A,
                'B' : B,
                "A_paths": sample[nameA],
                "B_paths": sample[nameB]
                }
    

    def __len__(self):
        return len(self.t2w_paths)

    def name(self):
        return 'ProstateDataset'

if __name__ == '__main__':
    #test
    n = ProstateDataset()
    n.initialize("/mnt/work/datasets/FLUTE/PICAI/seq-128x128x32_train")
    print(len(n))
    print(n[0])
    print(n[0]['A'].size())
