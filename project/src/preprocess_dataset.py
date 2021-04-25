# -*- coding: utf-8 -*-

import pydicom
import pandas as pd
from glob import glob
from tqdm import tqdm_notebook as tqdm
from sklearn.model_selection import train_test_split
import pickle

val_size = 0.10
train_file = '/content/drive/Mydrive/kaggle/siim/dicom-images-train/*/*/*.dcm'
rle_file = '/content/drive/Mydrive/kaggle/siim/train_rle.csv'
seed = 1994


def dicom_to_dict(dicom_data, file_path, rles_df, encoded_pixels=True):
    data = {}
    
    data['patient_name'] = dicom_data.PatientName
    data['patient_id'] = dicom_data.PatientID
    data['patient_age'] = int(dicom_data.PatientAge)
    data['patient_sex'] = dicom_data.PatientSex
    data['pixel_spacing'] = dicom_data.PixelSpacing
    data['file_path'] = file_path
    data['id'] = dicom_data.SOPInstanceUID
    
    if encoded_pixels:
        encoded_pixels_list = rles_df[rles_df['ImageId']==dicom_data.SOPInstanceUID]['EncodedPixels'].values
       
        pneumothorax = False
        for encoded_pixels in encoded_pixels_list:
            if encoded_pixels != ' -1' and encoded_pixels != '-1':
                pneumothorax = True
        
        data['encoded_pixels_list'] = encoded_pixels_list
        data['has_pneumothorax'] = pneumothorax
        data['encoded_pixels_count'] = len(encoded_pixels_list)
        
    return data


def parse_dicom(data_dcm, train_rle, encoded_pixels=True):
    data_metadata_df = pd.DataFrame()
    data_metadata_list = []
    for file_path in tqdm(data_dcm):
        dicom_data = pydicom.dcmread(file_path)
        train_metadata = dicom_to_dict(dicom_data, file_path, train_rle, encoded_pixels)
        data_metadata_list.append(train_metadata)
    data_metadata_df = pd.DataFrame(data_metadata_list)
    
    return data_metadata_df


def prepare_data(train_file, rle_file, val_size, seed):
    
    train_dcm = sorted(glob(train_file))
    print("Number of training files" , len(train_dcm))
    
    train_rle = pd.read_csv(rle_file)
    
    train_metadata_df  = parse_dicom(train_dcm, train_rle)
    
    masks = {}
    for index, row in train_metadata_df[train_metadata_df['has_pneumothorax']==1].iterrows():
        masks[row['id']] = list(row['encoded_pixels_list'])
    
    X_train, X_val, y_train, y_val = train_test_split(train_metadata_df.index, train_metadata_df['has_pneumothorax'].values, test_size=val_size, random_state=seed)
    X_train, X_val = train_metadata_df.loc[X_train]['file_path'].values, train_metadata_df.loc[X_val]['file_path'].values
    
    return X_train, X_val, masks
	
	
if __name__ == '__main__':
    
    X_train, X_val, masks = prepare_data(train_file, rle_file, val_size, seed)
    
    train_output = open('process_data/X_train.pkl', 'wb')
    pickle.dump(X_train, train_output)
    train_output.close()
    
    val_output = open('process_data/X_val.pkl', 'wb')
    pickle.dump(X_val, val_output)
    val_output.close()
    
    masks_output = open('process_data/masks.pkl', 'wb')
    pickle.dump(masks, masks_output)
    masks_output.close()