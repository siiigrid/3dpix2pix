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

picai_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda img: scale_and_normalize(img, 0, 99.75, 0, 1))
])


class ProstateDataset2(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.transform = picai_transforms

        # check what domain we want and load those images specificallyç
        # change root for this: /home/usuaris/imatge/sigrid.vila/files/projects/3d-pix2pix-CycleGAN/generated_att

        directory = os.path.join(self.root, opt.name[12:15] + "2" + opt.name[8:11])
        print("soc a prostatw2")

        print(directory)
        self.data = []

        self.source_paths = sorted(glob(os.path.join(directory, '*.npy')))
        print(len(self.source_paths))


        print(self.opt.name[12:15])

        if self.opt.name[12:15] == "t2w":
            print("here")
            self.target_paths = sorted(glob(os.path.join("/mnt/work/datasets/FLUTE/PICAI/seq-128x128x32_test", '*t2w.nii.gz')))
        if self.opt.name[12:15] == "adc":
            self.target_paths = sorted(glob(os.path.join("/mnt/work/datasets/FLUTE/PICAI/seq-128x128x32_test", '*adc.nii.gz')))
        if self.opt.name[12:15] == "hbv":
            self.target_paths = sorted(glob(os.path.join("/mnt/work/datasets/FLUTE/PICAI/seq-128x128x32_test", '*hbv.nii.gz')))
        print(len(self.target_paths))



        for src,trg in zip(self.source_paths,self.target_paths):
            subject = os.path.basename(src)[:-8]
            self.data.append({"source": src, "target": trg, "subject_id": subject})



    def __getitem__(self, index):
        #returns samples of dimension [channels, z, x, y]
        sample = self.data[index]
        nameA = self.opt.name[8:11]
        nameB = self.opt.name[12:15]

        nifti_data_B = nib.load(sample["target"])
        B = nifti_data_B.get_fdata()
        B = self.transform(B).unsqueeze(0)



        A = np.load(sample["source"])
        A = torch.from_numpy(A)


        return {
                'A' : A,
                "A_paths": sample["source"],
                'B' : B,
                "B_paths": sample["target"],
                }
    

    def __len__(self):
        return len(self.source_paths)

    def name(self):
        return 'ProstateDataset2'

if __name__ == '__main__':
    #test
    n = ProstateDataset2()
    n.initialize("/home/usuaris/imatge/sigrid.vila/files/projects/3d-pix2pix-CycleGAN/generated_att")
    print(len(n))
    print(n[0])
    print(n[0]['A'].size())
