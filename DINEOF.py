'''
Author: your name
Date: 2021-07-21 09:31:50
LastEditTime: 2021-07-21 19:18:10
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /DINEOF_GPU/DINEOF.py
'''
import torch
import numpy as np
from torch import unsqueeze
import xarray as xr
import scipy

def load_data():
    nc = xr.open_dataset('/data/Chenjq/Southsea_OCCCI/OCCCI/CCI_ALL-v5.0-DAILY.nc')
    data = np.transpose(nc.chlor_a.data,(1,2,0))

    num_lat = data.shape[0]
    num_lon = data.shape[1]
    num_time = data.shape[2]

    # 1 参与运算  0 不参与运算
    mask = np.ones((num_lat,num_lon))
    for la in range(num_lat):
        for lo in range(num_lon):
            if np.sum(np.isnan(data[la,lo,:]))>int(0.9*num_time):
                mask[la,lo] = 0 
    return data,mask

def dineof_gpu(data,mask,Max_EOF=3,rms_delta = 0.1):
    '''
    input:  data (lat,lon,time) 
            mask (lat,lon)
            Max_EOF 最大的EOF模态
            rms_delta 最小的误差阈值
    '''
    x = data[mask==1] # 剔除掉陆地
    
    # 得到确实数据的索引
    index = np.arange(len(x.reshape(-1)))  
    nan_idx = np.where(np.isnan(x.reshape(-1)))
    
    # 交叉验证索引
    val_idx_random = np.random.choice(index,int(0.1*len(index)))
    val_idx = np.concatenate([val_idx_random,nan_idx[0]])
    
    # 将缺失值替换为 0 
    x[np.isnan(x)]=0 
    
    # 初始化一些参属
    eof_n = 0
    RMS = []
    rms_prev = np.inf
    perform = []
    rms_now = 0
    while((rms_prev - rms_now > rms_delta)&(eof_n<=Max_EOF)):
        rms_prev = rms_now
        
        xx = torch.from_numpy(x).cuda()
        U,S,V = torch.svd(xx)
        Reci = torch.mm(torch.mul(unsqueeze(U[:,eof_n],dim=1),S[eof_n]),unsqueeze(V[:,eof_n],dim=1).T)
        Reci = Reci.cpu()
        # 需要释放GPU
        torch.cuda.empty_cache()
        rms_now = np.sqrt(np.nanmean(Reci.reshape(-1)[val_idx]-x[val_idx])^2)
        RMS.append(rms_now)
        perform.append((eof_n,rms_now))
        print((eof_n,rms_now))
        
        if(rms_now==min(RMS)):
            data[mask==1]  = Reci
            print("done")
            perform_best = (eof_n,rms_now)
    return data,perform,perform_best

if __name__=='__main__':
    data,mask = load_data()
    data,perform,perform_best = dineof_gpu(data,mask)