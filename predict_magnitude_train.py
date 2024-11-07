import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import gc
from torch.utils.data import Dataset, DataLoader, random_split
from utils.utils import *
from model.model_seismic_clip import *
from tqdm.auto import tqdm
import math
from colorama import init


metadata_path = '../earthquake_dataset/tsmip/metadata.csv'
waveform_path = '../earthquake_dataset/tsmip/waveforms.hdf5'
spec_encoder_path = 'Pretrain/pretrained_models/spec_encoder.ckpt'
predict_model_path = 'Pretrain/predict_magnitude/predict_model3.ckpt'
loss_csv_path = "Pretrain/predict_magnitude/val_loss3.csv"
predict_csv_path = "Pretrain/predict_magnitude/predict3.csv"

infoLabel = [
    'source_magnitude'
    ]

unused_key = [
    'data',
    'data_format',
    'data_format/component_order',
    'data_format/dimension_order',
    'data_format/measurement',
    'data_format/sampling_rate',
    'data_format/unit'
]


# define read waveform_file function
def read_hdf5(waveform_path, unused_key):
    f = h5py.File(waveform_path, 'r')
    keys = []
    f.visit(keys.append)
    for i in unused_key:
        keys.remove(i)
    return f, keys

# define norm and spectrum function
def cal_norm_spectrogram(x,window_length,nfft,sample_rate):
    spec = np.zeros([x.shape[0],3,int(x.shape[-1]/window_length * 2),int(nfft/2)])
    for n in range(x.shape[0]):
        for i in range(3):
            _, _, spectrogram = stft(x[n,i,:], fs=sample_rate, window='hann', nperseg=window_length, noverlap=int(window_length/2), nfft=nfft,boundary='zeros')
            spectrogram = spectrogram[1:,1:]
            spec[n,i,:] = np.abs(spectrogram).transpose(1,0)
    return spec
def norm(x):
    data = x.copy()
    for i in range(x.shape[1]):
        for j in range(x.shape[2]):
            if data[:,i,j].std() == 0:
                data[:,i,j] = (data[:,i,j] - data[:,i,j].mean())
            else:
                data[:,i,j] = (data[:,i,j] - data[:,i,j].mean())/data[:,i,j].std()
    return data

# define a function for select spectrum
def select_part_spec(t1,t2,data,dt=500,window_length = 8, nfft = 100, sample_rate=500):
#t1 and t2 is the start and end time, dt is sampling rate, each data has two stations? Here we set 1 sample with 1 minute. 
    select_data = np.zeros([500,2*(t2-t1),3])
    for i in range(t1,t2,1):
        if (i+1)*dt <= data.shape[0]:
            select_data[:,2*(i-t1):2*(i-t1+1),:] = data[i*dt:(i+1)*dt,:,:].copy()
        else:
            select_data[0:data.shape[0]-i*dt,2*(i-t1):2*(i-t1+1),:] = data[i*dt:data.shape[0],:,:].copy()
    select_data = norm(select_data).transpose(1,2,0)
    select_data = cal_norm_spectrogram(select_data,window_length,nfft,sample_rate)
    # the select data shape is [2(t2-t1),3,50,125], since we need the data shape is [50,120], we only use the first 120 samples of each data.
    return select_data

def data_preprocessing(waveform_data):
    for i in range(len(waveform_data)):
        waveform_data[i]=np.transpose(waveform_data[i],(2, 0, 1))
    
    return waveform_data

def split_train_val_test(metadata_data):
    metadata_data_train_val = metadata_data[metadata_data["split"].isin(['train', 'dev'])]
    metadata_data_test = metadata_data[metadata_data["split"].isin(['test'])]

    return metadata_data_train_val, metadata_data_test

def read_data_and_info(f, metadata_data, keys, batch_size, bucket_state, waveform_state):
    waveform_data = []
    info_trace_name = []

    if bucket_state == '':
        bucket_state = keys[0]

    print(f"[Start] Bucket_state is {bucket_state} and Waveform_state is {waveform_state}")
    pbar = tqdm(total=batch_size, desc="read_data_and_info")

    while(len(waveform_data) < (batch_size)): # 讀取batch_size大小的資料
        temp_wave = []
        temp_trace_name = ''
        if np.ndim(f[bucket_state]) == 2:
            temp_trace_name = bucket_state[5:len(bucket_state)]
            if (temp_trace_name == metadata_data).any().any():
                temp_wave = f[bucket_state]
                temp_wave = np.expand_dims(temp_wave,axis=0)
                waveform_data.append(temp_wave)
                info_trace_name.append(temp_trace_name)
                pbar.set_postfix(trace_name = temp_trace_name)
                pbar.update(1)

            index = keys.index(bucket_state)
            if (index + 1) < len(keys):
                bucket_state = keys[index + 1]
                waveform_state = 0
                continue

            else:
                break

        elif np.ndim(f[bucket_state]) == 3:
            temp_trace_name = bucket_state[5:len(bucket_state)] + '$' + str(waveform_state) + ','
            if metadata_data["trace_name"].str.startswith(temp_trace_name).any():
                temp_wave = f[bucket_state][waveform_state]
                temp_wave = np.expand_dims(temp_wave,axis=0)
                waveform_data.append(temp_wave)
                info_trace_name.append(temp_trace_name)
                pbar.set_postfix(trace_name = temp_trace_name)
                pbar.update(1)

            if (f[bucket_state].shape[0] - 1) > waveform_state:
                waveform_state += 1

            else:
                index = keys.index(bucket_state)
                if (index + 1) < len(keys):
                    bucket_state = keys[index + 1]
                    waveform_state = 0
                    continue

                else:
                    break
        
    pbar.close()
    print(f"[End] Bucket_state is {bucket_state} and Waveform_state is {waveform_state}")                
    return waveform_data, info_trace_name, bucket_state, waveform_state

def ToSpectrum(waveform_data, total_trace_name, metadata_data):
    spec_data = []
    info = []
    label = []
    
    tqd = tqdm(range(len(waveform_data)), desc= f'ToSpectrum')
    for i in range(len(waveform_data)):
        #波型轉換成頻譜圖
        tqd.set_postfix(shape = waveform_data[i].shape)
        spec_data.append(select_part_spec(0, math.ceil(waveform_data[i].shape[0]*0.002), waveform_data[i]))

        info_trace_name = total_trace_name[i]
        text = []
        info.append(metadata_data[metadata_data["trace_name"].str.startswith(info_trace_name)])
        
        for j in range(len(infoLabel)):
            text.append(float(info[i][infoLabel[j]].values))

        label.append(text)
        
        tqd.update(1)
    tqd.close()
    return spec_data, info, label


class CustomDataset(Dataset):
    def __init__(self, audio, text):
        self.audio = audio
        self.text = text

    def __len__(self):
        return len(self.audio)

    def __getitem__(self, idx):
        return self.audio[idx], self.text[idx]

class Regression(nn.Module):
    def __init__(self):
        super(Regression, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(384, 1), 
            nn.ReLU()
        )
        '''
        self.net = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        '''
    
    def forward(self, x):
        return self.net(x).squeeze(1)
    
def main():

    init()

    metadata_data = pd.read_csv(metadata_path)
    f, keys = read_hdf5(waveform_path, unused_key)
    for k in keys:
        print(k)
    print("metadata_shape:", metadata_data.shape)
    print('key_len: ', len(keys))

    metadata_data, _ = split_train_val_test(metadata_data)

    batch_size = 128

    device = torch.device("cuda")

    current_bucket_state = ''
    current_waveform_state = 0
    first = True
    final = True

    total_val_loss = []

    # 訓練模型
    while(final):
        temp_output = torch.empty(0, 384).to(device)
        labeled_magnitude = []

        waveform_data, total_trace_name, current_bucket_state, current_waveform_state = read_data_and_info(
            f, metadata_data, keys, batch_size, current_bucket_state, current_waveform_state
        )
        
        if len(waveform_data) != batch_size:
            print(f'waveform_data.len = {len(waveform_data)} != {batch_size}')
            final = False

        print(f'waveform_data.len = {len(waveform_data)} info_trace_name.len = {len(total_trace_name)}')

        waveform_data = data_preprocessing(waveform_data)
        spec_data, info, label = ToSpectrum(waveform_data, total_trace_name, metadata_data)

        print('spec_data_length: ', len(spec_data))
        print('info_length: ', len(info))
        print('label_length: ', len(label))

        label = torch.tensor(label)
        print('label_torch_shape: ', label.shape)

        audio = torch.empty(0, 3, 120, 50).to(device)
        for i in range(len(spec_data)):
            temp = np.expand_dims(spec_data[i][0],axis=0)
            spec_data_torch = torch.tensor(temp[:,:,0:120,:]).to(device)
            audio = torch.cat((audio, spec_data_torch), dim=0)
        print('audio_torch_shape: ', audio.shape)
        
        DataSet = CustomDataset(audio, label)
        spec_loader = DataLoader(DataSet, batch_size=1, shuffle=False)

        del waveform_data, info, audio, label, spec_data, temp, spec_data_torch
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

        del model, audio, text, batch, spec_loader
        gc.collect()
        torch.cuda.empty_cache()

        labeled_magnitude_torch = torch.tensor(labeled_magnitude)

        dataset = CustomDataset(temp_output, labeled_magnitude_torch)

        val_rate = 0.2
        val_size = int(val_rate * len(DataSet))

        train_size = len(DataSet) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        del temp_output, labeled_magnitude, labeled_magnitude_torch, DataSet, dataset, train_dataset, val_dataset
        gc.collect()
        torch.cuda.empty_cache()

        predict_model = Regression().to(device)
        #predict_optimizer = optim.Adam(predict_model.parameters(), lr=1e-4)
        predict_optimizer = optim.Adam(predict_model.parameters(), lr=1e-5)
        predict_loss = nn.MSELoss()
        if first == False:
            predict_model.load_state_dict(torch.load(predict_model_path))
            print('load predict model')

        # 設定early_stop相關參數
        early_stop = 50
        best_loss = 100.0
        
        predict_epoch = 800
        for epoch in range(predict_epoch):
            predict_model.train()
            train_loss = []

            for batch in tqdm(train_loader):
                spec, label = batch

                logits = predict_model(spec.to(device))
                loss = predict_loss(logits, label.to(device))

                predict_optimizer.zero_grad()
                loss.backward(retain_graph=True)
                predict_optimizer.step()

                train_loss.append(loss.item())
            
            train_loss = sum(train_loss) / len(train_loss)
            print(f"[ Train | {epoch + 1:03d}/{predict_epoch:03d} ] loss = {train_loss:.5f}")

            del spec, label, batch, logits, loss, train_loss
            gc.collect()
            torch.cuda.empty_cache()

            predict_model.eval()
            valid_loss = []

            for batch in tqdm(val_loader):
                spec, label = batch

                with torch.no_grad():
                    logits = predict_model(spec.to(device))

                loss = predict_loss(logits, label.to(device))

                valid_loss.append(loss.item())

            valid_loss = sum(valid_loss) / len(valid_loss)
            print(f"[ Valid | {epoch + 1:03d}/{predict_epoch:03d} ] loss = {valid_loss:.5f}")

            del spec, label, batch, logits, loss
            gc.collect()
            torch.cuda.empty_cache()

            if valid_loss < best_loss:
                early_stop = 20
                best_loss = valid_loss
                # 儲存模型
                torch.save(predict_model.state_dict(), predict_model_path)
                print(f"saving predict_model with loss = {valid_loss:.5f}")

            else:
                early_stop -= 1

            # early stop
            if early_stop < 0:
                total_val_loss.append(best_loss)
                print("Early Stopping")
                break

        if early_stop >= 0:
            total_val_loss.append(best_loss)
        first = False

        del predict_model, predict_optimizer, predict_loss, train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache()

    with open(loss_csv_path, "w") as f:

        f.write("Id, loss\n")
        for i in range(len(total_val_loss)):
            f.write(f"{i}, {total_val_loss[i]}\n")

    print(str(loss_csv_path), "written")

if __name__ == "__main__":
    main()