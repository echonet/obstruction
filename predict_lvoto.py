import torch 
import argparse
import numpy as np
import pandas as pd
from pathlib import Path 
import tqdm
import torch.nn.functional  as F
from torch.utils.data import DataLoader
from utils import load_model,dicom_to_tensor


'''
    This script takes in a user-specified path to a directory of A4C DICOMs and
    predicts the presence of left ventricular outflow tract obstruction. Predictions 
    are saved as a csv file in the directory that the script is run from. 
    The predictions file has 3 columns: 
        1. filename - name of the DICOM file 
        2. preds - output from the deep learning model 
        3. pred_lvoto - binary label where 0 indicates no obstruction and 1 indicates
        LVOT obstruction; label is determined using a Youden's index of 0.503
'''

parser = argparse.ArgumentParser()
parser.add_argument('--data_path',type=str,required=True)
args = parser.parse_args()

data_path = Path(args.data_path)
if not data_path.exists():
    print('File not found. Please enter path to an existing directory')
else: 
    ### Create Pytorch dataset from user-inputted directory of DICOM files
    dcm_stack = []
    files = []
    ### Convert each DICOM to tensor as input to model
    for dcm_path in data_path.iterdir(): 
        input = dicom_to_tensor(dcm_path)
        dcm_stack.append(input)
        files.append(dcm_path)
    input = torch.stack(dcm_stack)
    labels = torch.zeros(input.shape[0])
    dataset = torch.utils.data.TensorDataset(input,labels)
    print(f'Converted DICOM files to tensors:\t{len(dataset)}')
    ### Set up dataloader
    batch_size = 24
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=False)
    ### Load model 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = load_model(device)
    preds = []
    ### Run inference on each batch
    for idx,(batch,label) in tqdm.tqdm(enumerate(dataloader)):
        batch = batch.to(device)
        pred_lvoto = model(batch)
        start = idx*batch_size
        end = min((idx+1)*batch_size,len(dataset))
        # preds[start:end] = F.sigmoid(pred_lvoto).detach().cpu().numpy()
        preds.append(pred_lvoto.detach().cpu().numpy())
    preds = np.vstack(preds)
    preds = preds.ravel() # Convert to 1D array
    ### Create dataframe with columns for filename and predictions
    predicted_lvoto = pd.DataFrame(
        {
            'filename':files,
            'preds':preds
        }
    )
    ### Activation function for predictions 
    predicted_lvoto['preds'] = F.sigmoid(torch.tensor(predicted_lvoto['preds'])).numpy()
    ### Threshold predictions into binary labels 
    youden = 0.503422517795
    predicted_lvoto['pred_lvoto'] = np.where(predicted_lvoto['preds']>=youden,1,0)
    ### Save predictions to file 
    predicted_lvoto.to_csv('predictions.csv',index=None)

        

