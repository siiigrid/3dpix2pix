import os.path
import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
from glob import glob
import nibabel as nib

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

picai_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        #transforms.ScaleIntensityRangePercentilesd(lower=0, upper=99.75, b_min=0, b_max=1)
    ]
)


class ProstateDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        self.t2w_paths = sorted(glob(os.path.join(self.root, '*t2w.nii.gz')))
        self.adc_paths = sorted(glob(os.path.join(self.root, '*adc.nii.gz')))
        self.hbv_paths = sorted(glob(os.path.join(self.root, '*hbv.nii.gz')))
        print(len(self.t2w_paths),len(self.adc_paths),len(self.hbv_paths), self.root)
        
        
        self.data = []

        if len(self.t2w_paths) == 1400: self.seg_paths = sorted(glob(os.path.join('/mnt/gpid08/datasets/FLUTE/PICAI/mask_combined/binary_masks/training_set', '*.nii.gz')))
        else:  self.seg_paths = sorted(glob(os.path.join('/mnt/gpid08/datasets/FLUTE/PICAI/mask_combined/binary_masks/test_set', '*.nii.gz')))

        self.transform = picai_transforms

        for t2w, adc, hbv, seg in zip(self.t2w_paths, self.adc_paths, self.hbv_paths, self.seg_paths):
            subject = os.path.basename(t2w)[:-11]
            assert subject == os.path.basename(adc)[:-11]
            assert subject == os.path.basename(hbv)[:-11]
            assert subject == os.path.basename(seg)[:-7]
            self.data.append({"adc":adc, "t2w":t2w, "hbv": hbv,"subject_id": subject})
        


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
