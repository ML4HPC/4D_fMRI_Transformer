import os
import numpy as np
import nibabel as nib
import torch
from multiprocessing import Process, Queue

# Stella - 이게 HCP 데이터 전체에 통용되는 전처리 방법인가?
def read_abcd(file_path,global_norm_path,per_voxel_norm_path, count,queue=None):
    ## remove former 20 timepoints (이미 4.cleaned image는 10개 지워진 상태이긴 함)
    img_orig = torch.from_numpy(np.asanyarray(nib.load(file_path).dataobj)[10:-10, 10:-10, 0:-10, 20:]).to(dtype=torch.float32) # (x,y,z,timepoint)
    background = img_orig == 0
    img_temp = (img_orig - img_orig[~background].mean()) / (img_orig[~background].std()) #global normalization
    img = torch.empty(img_orig.shape)
    img[background] = img_temp.min()
    img[~background] = img_temp[~background]
    img = torch.split(img, 1, 3) #timepoint(3)을 기준으로 1토막씩 나누어 튜플로 반환 (각각이 3차원 이미지)
    for i, TR in enumerate(img):
        torch.save(TR.clone(),
                   os.path.join(global_norm_path, 'rfMRI_TR_' + str(i) + '.pt'))
    
    # repeat for per voxel normalization
    img_temp = (img_orig - img_orig.mean(dim=3, keepdims=True)) / (img_orig.std(dim=3, keepdims=True))
    img = torch.empty(img_orig.shape)
    img[background] = img_temp.min()
    img[~background] = img_temp[~background]
    img = torch.split(img, 1, 3) #timepoint(3)을 기준으로 1토막씩 나누어 튜플로 반환 (각각이 3차원 이미지. 즉 4차원 데이터를 시간으로 쪼개서 별도의 파일로 저장하는거.)
    for i, TR in enumerate(img):
        torch.save(TR.clone(),
                   os.path.join(per_voxel_norm_path, 'rfMRI_TR_' + str(i) + '.pt'))
    print('finished another subject. count is now {}'.format(count))

def main():
    abcd_path = '/pscratch/sd/j/junbeom/ABCDfMRI/4.cleaned_image'
    save_path = '/pscratch/sd/s/stella/ABCD_TFF_20_timepoint_removed'
    os.makedirs(save_path, exist_ok=True)
    #all_files_path = os.path.join(hcp_path,'data')
    queue = Queue()
    count = 0
    subj_list = sorted(os.listdir(abcd_path))
    # f = open('/global/cfs/cdirs/m3898/HCP1200_TFF/norm_left.txt')
    # subj_list = f.read().splitlines()
    # f.close()
    # for subj in os.listdir(all_files_path):
    for file_name in subj_list[:]:
        if 'nii.gz' in file_name: #remove ipynb_checkpoint and Untitled.ipynb
            subj_path = os.path.join(abcd_path,file_name)
            subj = file_name.split('-')[1].split('.')[0]
            print(subj)
            try:
                #file_path = os.path.join(subj_path,os.listdir(subj_path)[0])
                #hand = file_path[file_path.find('REST1_')+6:file_path.find('.nii')]
                global_norm_path = os.path.join(save_path, 'MNI_to_TRs',subj,'global_normalize')
                per_vox_norm_path = os.path.join(save_path, 'MNI_to_TRs', subj, 'per_voxel_normalize')
                os.makedirs(global_norm_path, exist_ok=True)
                os.makedirs(per_vox_norm_path, exist_ok=True)
                count += 1
                print('start working on subject '+ subj)
                p = Process(target=read_abcd, args=(subj_path,global_norm_path,per_vox_norm_path, count, queue)) ### 이거 원래 hand 있었거든? 없는 걸로 수정하자.
                p.start()
                if count % 4 == 0:
                    p.join()  # this blocks until the process terminates
            except Exception:
                print('encountered problem with '+subj)
                print(Exception)
if __name__ == '__main__':
    main()
