import torch 
import numpy as np
from pathlib import Path 
from torch.utils.data import Dataset
import torchvision
import pydicom


def load_model(device, weights_path='model_best_epoch_val_loss.pt'):
    model = torchvision.models.video.r2plus1d_18(num_classes=1)
    weights = torch.load(weights_path,map_location=device)
    weights = {key[2:]:val for key,val in weights.items()}
    print('Model loaded:\n', model.load_state_dict(weights))
    model.to(device)
    return model.eval()

def get_ybr_to_rgb_lut(filepath,save_lut=False):
    global _ybr_to_rgb_lut
    _ybr_to_rgb_lut = None
    # return lut if already exists
    if _ybr_to_rgb_lut is not None:
        return _ybr_to_rgb_lut
    # try loading from file
    lut_path = Path(filepath).parent / 'ybr_to_rgb_lut.npy'
    if lut_path.is_file():
        _ybr_to_rgb_lut = np.load(lut_path)
        return _ybr_to_rgb_lut
    # else generate lut
    a = np.arange(2 ** 8, dtype=np.uint8)
    ybr = np.concatenate(np.broadcast_arrays(a[:, None, None, None], a[None, :, None, None], a[None, None, :, None]), axis=-1)
    _ybr_to_rgb_lut = pydicom.pixel_data_handlers.util.convert_color_space(ybr, 'YBR_FULL', 'RGB')
    if save_lut:
        np.save(lut_path, _ybr_to_rgb_lut)
    return _ybr_to_rgb_lut

def ybr_to_rgb(filepath,pixels: np.array):
    lut = get_ybr_to_rgb_lut(filepath)
    return lut[pixels[..., 0], pixels[..., 1], pixels[..., 2]]

def change_doppler_color(dcm_path):
    ds = pydicom.dcmread(dcm_path)
    pixels = ds.pixel_array
    if ds.PhotometricInterpretation == 'MONOCHROME2':
        input_image = np.stack((pixels,)*3,axis=-1)
    elif ds.PhotometricInterpretation in ['YBR_FULL',"YBR_FULL_422"]:
        input_image = ybr_to_rgb(dcm_path,pixels)
    elif ds.PhotometricInterpretation == "RGB": 
        pass
    else:
        print("Unsupported Photometric Interpretation: ",ds.PhotometricInterpretation)
    return input_image 

def dicom_to_tensor(dcm_path,n=112,frames=32):
    ### Color convert DICOM to array
    pixels = change_doppler_color(dcm_path)
    ### Convert array to tensor & reformat
    pixels = torch.from_numpy(pixels)
    pixels = pixels.permute(-1,0,1,2) # Dims now C,F,H,W
    ### Resize array to (112,112)
    resizer = torchvision.transforms.Resize((n,n))
    resized_input = resizer(pixels)
    ### Read in first 32 frames with sample rate of 2
    c,f,h,w = resized_input.shape
    ### If DICOM has < 32 frames, return tensor of 0s 
    if f < frames: 
        return torch.zeroes((3,int(frames/2),n,n))
    resized_input = resized_input[:,np.arange(0,frames,2),:,:]
    ### Normalize 
    norm_input = resized_input/255. 
    input = norm_input.to(torch.float32)
    return input
