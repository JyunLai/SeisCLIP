import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import h5py
import gc
from torch.utils.data import DataLoader
from utils.utils import *
from model.model_seismic_clip import *
from tqdm.auto import tqdm
from colorama import init
import predict_magnitude_train


metadata_path = predict_magnitude_train.metadata_path
waveform_path = predict_magnitude_train.waveform_path
spec_encoder_path = predict_magnitude_train.spec_encoder_path
predict_model_path = predict_magnitude_train.predict_model_path
predict_csv_path = predict_magnitude_train.predict_csv_path


def read_test_data_and_info(f, metadata_data, keys, location):
    waveform_data = []
    label = []
    total_trace_name = []
    start_pos = location * batch_size
    if location == 68:
        end_pos = 8712
    else:
        end_pos = (location + 1) * batch_size

    pbar = tqdm(total=batch_size, desc="read_test_data_and_info")
    for key in range(start_pos, end_pos):
        string = keys[key]
        loc_1 = 0
        loc_2 = 0
        index = 0
        while(loc_1 == 0 or loc_2 == 0):
            if string[index] == '$':
                loc_1 = index

            elif string[index] == ',':
                loc_2 = index

            index += 1

        bucket_state = string[:loc_1]
        waveform_state = string[loc_1+1:loc_2]

        temp_wave = []
        temp_trace_name = bucket_state + '$' + waveform_state + ','
        if metadata_data["trace_name"].str.startswith(temp_trace_name).any():
            bucket_state = 'data/' + str(bucket_state)
            temp_wave = f[bucket_state][int(waveform_state)]
            temp_wave = np.expand_dims(temp_wave,axis=0)
            waveform_data.append(temp_wave)
            info = metadata_data[metadata_data["trace_name"].str.startswith(temp_trace_name)]
            label.append(float(info[predict_magnitude_train.infoLabel[0]]))
            total_trace_name.append(temp_trace_name[:-1])
            pbar.update(1)

    del temp_wave
    gc.collect()
    torch.cuda.empty_cache()
    return waveform_data, label, total_trace_name


init()

metadata_data = pd.read_csv(metadata_path)
f, keys = predict_magnitude_train.read_hdf5(waveform_path, predict_magnitude_train.unused_key)
for k in keys:
    print(k)
print("metadata_shape:", metadata_data.shape)
print('key_len: ', len(keys))

metadata_data, metadata_data_test = predict_magnitude_train.split_train_val_test(metadata_data)

batch_size = 128

device = torch.device("cuda")

# Testing
test_keys = metadata_data_test['trace_name'].tolist()
test_id = []
test_pred = torch.empty((0))
test_correct = torch.empty((0))

for i in range(69):
    temp_output = torch.empty(0, 384).to(device)
    labeled_magnitude = []

    waveform_data, label, trace_name = read_test_data_and_info(f, metadata_data_test, test_keys, i)
    waveform_data = predict_magnitude_train.data_preprocessing(waveform_data)
    spec_data = predict_magnitude_train.ToSpectrum(waveform_data)

    print('test_spec_data_length: ', len(spec_data))
    print('test_label_length: ', len(label))

    test_id += trace_name

    label = torch.tensor(label)
    print('test_label_torch_shape: ', label.shape)

    audio = torch.empty(0, 3, 120, 50).to(device)
    for j in range(len(spec_data)):
        temp = np.expand_dims(spec_data[j][0],axis=0)
        spec_data_torch = torch.tensor(temp[:,:,0:120,:]).to(device)
        audio = torch.cat((audio, spec_data_torch), dim=0)
    print('test_audio_torch_shape: ', audio.shape)
    
    DataSet = predict_magnitude_train.CustomDataset(audio, label)
    spec_loader = DataLoader(DataSet, batch_size=1, shuffle=False)

    del waveform_data, audio, label, spec_data, temp, spec_data_torch
    gc.collect()
    torch.cuda.empty_cache()

    model = AUDIO_CLIP(
        embed_dim = 384,
        text_input = 7,
        text_width = batch_size,
        text_layers = 2,
        spec_model_size = 'small224'
        ,device_name = device
    ).to(device)
    model.load_state_dict(torch.load(spec_encoder_path))
    print('load spec_encoder')

    model.eval()
    for batch in tqdm(spec_loader):
        audio, text = batch

        temp_output = torch.cat((temp_output, model.encode_audio(audio)), dim=0)
        labeled_magnitude.append(text)

    del model, audio, text, batch, spec_loader, DataSet
    gc.collect()
    torch.cuda.empty_cache()

    labeled_magnitude_torch = torch.tensor(labeled_magnitude)

    dataset = predict_magnitude_train.CustomDataset(temp_output, labeled_magnitude_torch)

    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    del temp_output, labeled_magnitude, labeled_magnitude_torch, dataset
    gc.collect()
    torch.cuda.empty_cache()

    predict_model = predict_magnitude_train.Regression().to(device)
    predict_model.load_state_dict(torch.load(predict_model_path))
    predict_loss = nn.MSELoss()

    test_loss = []
    for batch in tqdm(test_loader):
        spec, label = batch

        with torch.no_grad():
            logits = predict_model(spec.to(device))

        loss = predict_loss(logits, label.to(device))

        test_loss.append(loss.item())

        test_loss = sum(test_loss) / len(test_loss)
        print(f"[ Test {i} ] loss = {test_loss:.5f}")
        test_pred = torch.cat((test_pred, logits.cpu()), 0)
        test_correct = torch.cat((test_correct, label.cpu()), 0)
        
        del spec, label, batch, logits, loss
        gc.collect()
        torch.cuda.empty_cache()

    del predict_model, predict_loss, test_loader
    gc.collect()
    torch.cuda.empty_cache()

total_loss = nn.MSELoss()(test_pred.to(device), test_correct.to(device))
print("Total_Loss =", total_loss)

with open(predict_csv_path, "w") as f:

    f.write("Id, pred, correct\n")
    for i in range(len(test_id)):
        f.write(f"{test_id[i]}, {test_pred[i]}, {test_correct[i].item()}\n")