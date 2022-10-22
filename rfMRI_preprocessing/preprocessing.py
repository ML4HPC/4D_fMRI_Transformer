from monai.transforms import LoadImage
import torch
import os
import time

start_time = time.time()

load_root = '/storage/bigdata/hcp1200/data_raw_rsfMRI'
save_root = '/storage/bigdata/hcp1200/data_raw_MNI_to_TRs'
dirs = os.listdir(load_root)

for dir in sorted(dirs):
    print("processing: " + dir, flush=True)
    path = os.path.join(load_root, dir,"rfMRI_REST1_LR.nii.gz")
    try:
        data, meta = LoadImage()(path)
    except:
        continue

    save_dir = os.path.join(save_root,dir)
    isExist = os.path.exists(save_dir)
    if not isExist:
        os.makedirs(save_dir)
    data = torch.tensor(data)[8:-8, 8:-8, :-10, 10:-10]
    #data = data.as_tensor()[8:-8, 8:-8, :-10, 10:-10]
    background = data==0

    global_mean = data[~background].mean()
    global_std = data[~background].std()
    data_temp = (data - global_mean) / global_std

    data_global = torch.empty(data.shape)
    data_global[background] = data_temp.min()
    data_global[~background] = data_temp[~background]

    data_global = data_global.type(torch.float16)

    data_global_split = torch.split(data_global, 1, 3)
    for i, TR in enumerate(data_global_split):
        torch.save(TR.clone(), os.path.join(save_dir,"frame_"+str(i)+".pt"))

    # Note that this code is different from the original code
    voxel_mean = data_global.mean(dim=3, keepdim = True)
    voxel_std = data_global.std(dim=3, keepdim = True)
    
    # The following commented lines are the same as the original preprocessing code
    # voxel_mean = data.mean(dim=3, keepdim = True)
    # voxel_std = data.std(dim=3, keepdim = True)

    voxel_mean = voxel_mean.type(torch.float16)
    voxel_std = voxel_std.type(torch.float16)

    #os.makedirs(save_dir)
    
    torch.save(voxel_mean.detach().contiguous(), os.path.join(save_dir,"voxel_mean.pt"))
    torch.save(voxel_std.detach().contiguous(), os.path.join(save_dir,"voxel_std.pt"))

end_time = time.time()
print('\nTotal', round((end_time - start_time) / 60), 'minutes elapsed.')