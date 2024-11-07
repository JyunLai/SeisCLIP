import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import gc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import Dataset, DataLoader, random_split
from utils.utils import *
from model.model_seismic_clip import *
from tqdm.auto import tqdm
import matplotlib
import matplotlib.pyplot as plt
import math
from PIL import Image
from colorama import init, Fore, Style
#matplotlib.use("Agg")

metadata_path = '../earthquake_dataset/cwbsn/metadata.csv'
waveform_path = '../earthquake_dataset/cwbsn/waveforms.hdf5'
pretrain_model_path = 'Pretrain/pretrained_models/pretrain_model_50_120.pt'


infoLabel = [
    'trace_p_arrival_sample',
    'trace_p_weight',
    'path_p_travel_s',
    'trace_s_arrival_sample',
    'trace_s_weight',
    'path_ep_distance_km',
    'path_back_azimuth_deg'
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

def plot_spec(data,i):
    plt.imshow(data[i,0,:].T)
    plt.ylim(0,49)
    plt.yticks([])
    plt.show()

def data_preprocessing(waveform_data):
    for i in range(len(waveform_data)):
        waveform_data[i]=np.transpose(waveform_data[i],(2, 0, 1))
    
    return waveform_data

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(384, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CustomDataset(Dataset):
    def __init__(self, audio, text):
        self.audio = audio
        self.text = text

    def __len__(self):
        return len(self.audio)

    def __getitem__(self, idx):
        return self.audio[idx], self.text[idx]

def read_data_and_info(f, metadata_data, keys, batch_size, bucket_state, waveform_state):
    waveform_data = []
    info_trace_name = []

    if bucket_state == '':
        bucket_state = keys[0]

    print(f"[Start] Bucket_state is [{bucket_state}] and Waveform_state is [{waveform_state}]")
    pbar = tqdm(total=batch_size * 4, desc="read_data_and_info")

    while(len(waveform_data) < (batch_size * 4)): # 讀取2倍batch_size的資料
        temp_wave = []
        temp_trace_name = ''
        if np.ndim(f[bucket_state])==2:
            temp_trace_name = bucket_state[5:len(bucket_state)]
            if (temp_trace_name == metadata_data).any().any():
                temp_wave = f[bucket_state]
                temp_wave=np.expand_dims(temp_wave,axis=0)
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
        elif np.ndim(f[bucket_state])==3:
            temp_trace_name = bucket_state[5:len(bucket_state)] + '$' + str(waveform_state) + ','
            if metadata_data["trace_name"].str.startswith(temp_trace_name).any():

                info = metadata_data[metadata_data["trace_name"].str.startswith(temp_trace_name)]
                info_available = True
                for i in range(len(infoLabel)):
                    if np.isnan(info[infoLabel[i]]).any():
                        info_available = False

                if info_available:
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

init()

metadata_data = pd.read_csv(metadata_path)
f, keys = read_hdf5(waveform_path, unused_key)
for k in keys:
    print(k)
print("metadata_shape:", metadata_data.shape)
print('key_len: ', len(keys)) # 137

num_epochs = 500
batch_size = 64

device = torch.device("cuda")

model_path = 'Pretrain/pretrained_models/spec_encoder_cwbsn.ckpt'


# 設定一些基本參數
best_acc = 0.0
best_loss = 100.0

# 設定early_stop的標準
early_stop_standard = 40

current_bucket_state = ''
current_waveform_state = 0

IsFirstTime = True
Final = True
# 訓練模型
while(Final):
    waveform_data, total_trace_name, current_bucket_state, current_waveform_state = read_data_and_info(
        f, metadata_data, keys, batch_size, current_bucket_state, current_waveform_state
    )

    if len(waveform_data) != (4 * batch_size):
        print(f'waveform_data.len = {len(waveform_data)} != {4 * batch_size}')
        Final = False
    print(f'waveform_data.len = {len(waveform_data)} info_trace_name.len = {len(total_trace_name)}')

    waveform_data = data_preprocessing(waveform_data)
    spec_data, info, label = ToSpectrum(waveform_data, total_trace_name, metadata_data)
     
    print('spec_data_size: ', len(spec_data))
    print('info_size: ', len(info))
    print('label_size: ', len(label))

    label = torch.tensor(label)
    print('label.shape: ', label.shape)

    audio = torch.empty(0, 3, 120, 50).to(device)
    output = []
    for i in range(len(spec_data)):
        temp = np.expand_dims(spec_data[i][0],axis=0)
        spec_data_torch = torch.tensor(temp[:,:,0:120,:]).to(device)
        audio = torch.cat((audio, spec_data_torch), dim=0)


    print('audio.shape: ', audio.shape)
    
    DataSet = CustomDataset(audio, label) # 4 * batch_size

    torch.cuda.empty_cache()
    del spec_data

    val_rate = 0.25
    
    val_size = int(val_rate * len(DataSet)) # 1 * batch_size
    train_size = len(DataSet) - val_size # 3 * batch_size

    train_dataset, val_dataset = random_split(DataSet, [train_size, val_size])
    print('train_dataset_size: ', len(train_dataset))
    print('val_dataset_size: ', len(val_dataset))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    model = AUDIO_CLIP(
        embed_dim = 384,
        text_input = len(infoLabel),
        text_width = batch_size,
        text_layers = 2,
        spec_model_size = 'small224'
        ,device_name = device
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    if IsFirstTime == False:
        model.load_state_dict(torch.load(model_path))
        print('load model')

    early_stop = early_stop_standard # 40
    for epoch in range(num_epochs):
        model.train()
        train_loss = []
        train_accs = []

        for batch in tqdm(train_loader):
            audio, text = batch

            _, logits, loss = model(text.to(device), audio.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
            train_loss.append(loss.item())
            #train_accs.append(acc.item())

        print('train_loss_sum: ', sum(train_loss))
        train_loss = sum(train_loss) / len(train_loss)
        #train_acc = sum(train_accs) / len(train_accs)
        #print(f"[ Train | {epoch + 1:03d}/{num_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
        print(f"[ Train | {epoch + 1:03d}/{num_epochs:03d} ] loss = {train_loss:.5f}")


        model.eval()
        valid_loss = []
        valid_accs = []

        for batch in tqdm(val_loader):
            audio, text = batch
            with torch.no_grad():
                _, logits, loss = model(text.to(device), audio.to(device))
            #acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
            valid_loss.append(loss.item())
            #valid_accs.append(acc.item())

        valid_loss = sum(valid_loss) / len(valid_loss)
        #valid_acc = sum(valid_accs) / len(valid_accs)
        #print(f"[ Valid | {epoch + 1:03d}/{num_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
        print(f"[ Valid | {epoch + 1:03d}/{num_epochs:03d} ] loss = {valid_loss:.5f}")
        
        if valid_loss < best_loss:
            early_stop = early_stop_standard
            #best_acc = valid_acc
            best_loss = valid_loss
            torch.save(model.state_dict(), model_path)
            print(f"{Fore.RED}saving model with loss = {valid_loss:.5f}{Style.RESET_ALL}")
        else:
            early_stop -= 1

        if early_stop <= 0:
            print("Early Stopping")
            print(f"The best loss is {Fore.RED}{best_loss}{Style.RESET_ALL}")
            break
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    IsFirstTime = False