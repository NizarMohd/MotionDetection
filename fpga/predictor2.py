import pynq.lib.dma
from pynq import allocate
import numpy as np
from pynq import Overlay
import csv
import time
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

from scipy.fftpack import fft, fftfreq

import math
from queue import Queue
import csv
import os
class Predictor2:
   
    def __init__(self):
        W_S = 77
        self.ol = Overlay('mlp4002.bit')
        self.dma_send = [self.ol.axi_dma_0, self.ol.axi_dma_1]
        self.dma_recv = [self.ol.axi_dma_0, self.ol.axi_dma_1]
        self.input_buffer = [allocate(shape=(93,), dtype=np.uint32), allocate(shape=(93,), dtype=np.uint32)]
        self.output_buffer = [allocate(shape=(1,), dtype=np.uint32), allocate(shape=(1,), dtype=np.uint32)]
        self.window = [ [[],[],[],[],[],[]], [[],[],[],[],[],[]]]
      
        self.index =[0,0]
        self.allow_pred = [True, True]
       
#         #         optimal was 0
#         self.threshold = -1
        self.prev = [0, 0]
        #         optimal was [7,7,7,6]
        self.prediction_threshold = [ [7,7,7,6], [7, 7, 7,6]] #need double threshold for p2 on wed
     
        self.predicted_index = [0,0]
        self.timer = [0,0]
        self.window_size = W_S
        self.push_count = [0,0]
        
        #         optimal was 6
        self.interval_size = 7
        self.pred_count =[ [0,0,0,0,0], [0,0,0,0,0]]
        self.count = [0,0]
        
      
    def flush(self):
        W_S = 77
        self.window = [[],[],[],[],[],[]]
        self.index = 0
        self.prev = 0
        self.push_count = 0
        self.pred_count = [0,0,0,0,0]
        

           
    def preprocess(self, p):
        
        min_buf = []
        max_buf = []
        mean_buf = []
        median_buf = []
        mad_buf = []
        sd_buf = []
        iqr_buf = []
        entropy_buf = []
        rms_buf = []
        range_buf = []
        avg_magf_buf = []
        madf_buf = []
        tdiff = []
        i_min = []
        i_max = []
        corr_buf = []
        SAMPLE_RATE = 77
         
        for j in range(0, 6):
            for i in range(j + 1, 6):
                corr = np.corrcoef(self.window[p-1][j], self.window[p-1][i])
                corr_buf.append(corr[0, 1])
                
                    
        for j in range(0, 6):
           
            data = self.window[p-1][j]
           
            f_v = {}
            # each data is 1s window of one attr
            for item in data:
                if item in f_v:
                    f_v[item] = f_v[item] + 1
                else:
                    f_v[item] = 1
            N = len(data)

            min_buf.append(min(data))
            max_buf.append(max(data))
            i_min.append(np.argmin(data))
            i_max.append(np.argmax(data))
            range_buf.append(max(data) - min(data))

            # print("min: " + str(min_v))
            # print("max: " + str(max_v))

            mean = np.mean(data)
            mean_buf.append(mean)

            median = np.median(data)
            median_buf.append(median)

            q3, q1 = np.percentile(np.array(data), [75, 25])
            iqr = q3 - q1
            iqr_buf.append(iqr)

            n2 = 0
            n3 = 0
            n4 = 0
            h = 0
            mad = 0
            rms = 0

            for item in data:
                rms = rms + item ** 2
                n2 = (item - mean) ** 2 + n2
#                 n3 = (item - mean) ** 3 + n3
#                 n4 = (item - mean) ** 4 + n4
                mad = abs(item - mean) + mad
                p_i = f_v[item] / N

                h = p_i * math.log(p_i, 2) + h
            
            if N > 1:
                variance = n2 / (N - 1)
            else:
                variance = 0
            sd = math.sqrt(variance)
            sd_buf.append(sd)

            # print("sd: " + str(sd))
            mad = mad / N
            mad_buf.append(mad)

            rms = math.sqrt(rms / N)
            rms_buf.append(rms)

            entropy_buf.append(-1 * h)
#             skew = (n3 / N) / math.pow((n2 / N), 1.5)
#             skew_buf.append(skew)

#             kurt = (n4 / N) / (n2 / N) ** 2 - 3
#             kurt_buf.append(kurt)
            yf = fft(np.array(data))
            xf = fftfreq(N, 1 / SAMPLE_RATE)
            avg_magf = 0
            for i in range(len(xf)):
                avg_magf = np.abs(yf[i]) + avg_magf
            avg_magf = avg_magf / len(xf)
            avg_magf_buf.append(avg_magf)

            madf = 0
            for i in range(len(xf)):
                madf = madf + abs(np.abs(yf[i]) - mean)
            madf = madf / len(xf)
            madf_buf.append(madf)

        all_f = []
        for i in range(len(i_min)):
            tdiff.append(i_max[i] - i_min[i])

        for i in range(len(min_buf)):
            all_f.append(round(min_buf[i]))
            all_f.append(round(max_buf[i]))
            all_f.append(round(mean_buf[i]))
            all_f.append(round(median_buf[i]))
            all_f.append(round(mad_buf[i]))
            all_f.append(round(sd_buf[i]))
            all_f.append(round(iqr_buf[i]))
            all_f.append(round(entropy_buf[i]))
            all_f.append(round(rms_buf[i]))
            all_f.append(round(range_buf[i]))
            all_f.append(round(avg_magf_buf[i]))
            all_f.append(round(madf_buf[i]))
            all_f.append(round(tdiff[i]))
        for item in corr_buf:
            if math.isnan(item):
                item = 0
            all_f.append(round(item * 10 ** 2))
            
        return all_f


    def push(self, data, p):
 
        self.push_count[p-1] = self.push_count[p-1] + 1
        if len(data) == 6:
            if self.index[p-1] == self.window_size - 1:
                for i in range(len(data)):
                    self.window[p-1][i].pop(0)
                self.index[p-1] = self.index[p-1] - 1
           
            for i in range(len(data)):
                self.window[p-1][i].append(data[i])

            self.index[p-1] = self.index[p-1] + 1
            is_interval = self.push_count[p-1] % self.interval_size == self.interval_size - 1
            if is_interval and self.allow_pred[p-1]:
                features = self.preprocess(p)
                for i in range(len(features)):
                    self.input_buffer[p-1][i] = features[i]
          
                self.dma_send[p-1].sendchannel.transfer(self.input_buffer[p-1])
                self.dma_recv[p-1].recvchannel.transfer(self.output_buffer[p-1])
                self.dma_send[p-1].sendchannel.wait()
                self.dma_recv[p-1].recvchannel.wait()
                return self.output_buffer[p-1][0]
        return 0
   

    def send_to_dma(self, data):
        
        p = 1
        if self.allow_pred[p-1] == False:
            self.timer[p-1] = self.timer[p-1] + 1
            if self.timer[p-1] >= 77:
                self.allow_pred[p-1] = True
        pred = self.push(data, p)
        if pred > 0:
            if self.prev[p-1] == pred:
                    self.count[p-1] = self.count[p-1] + 1
                    self.pred_count[p-1][pred] = max(self.count[p-1], self.pred_count[p-1][pred])
            else:
                    self.count[p-1] = 0
            self.prev[p-1] = pred
            if self.pred_count[p-1][pred] >= self.prediction_threshold[p-1][pred-1]:
                self.pred_count[p-1] = [0,0,0,0,0]
                self.count[p-1] = 0
                self.timer[p-1] = 0
                self.allow_pred[p-1] = False
                return pred
      
        return 0
    def send_to_dma1(self, data):
        
        p = 2
        if self.allow_pred[p-1] == False:
            self.timer[p-1] = self.timer[p-1] + 1
            if self.timer[p-1] >= 77:
                self.allow_pred[p-1] = True
        pred = self.push(data, p)
        if pred > 0:
            if self.prev[p-1] == pred:
                    self.count[p-1] = self.count[p-1] + 1
                    self.pred_count[p-1][pred] = max(self.count[p-1], self.pred_count[p-1][pred])
            else:
                    self.count[p-1] = 0
            self.prev[p-1]= pred
            if self.pred_count[p-1][pred] >= self.prediction_threshold[p-1][pred-1]:
                self.pred_count[p-1] = [0,0,0,0,0]
                self.count[p-1] = 0
                self.timer[p-1] = 0
                self.allow_pred[p-1] = False
                return pred
        return 0

if __name__ == "__main__":
        f1
    
    
        
            
            
            
            
            
            
            
            
            
            
            

#          if pred > 0:
#                     if model.prev == pred:
#                         count = count + 1
#                     else:
#                         count = 0
#                         model.allow_pred = True
#                     if count == model.prediction_threshold:
#                         confusion_matrix[pred][k] = confusion_matrix[pred][k] + 1
#                         model.allow_pred = False
#                     model.prev = pred
                    
#             if model.allow_pred:
#                  confusion_matrix[0][k] = confusion_matrix[0][k] + 1 
                    
#             print(confusion_matrix, "avg time: ", str(end - start))                       
                        
                        
                  
                    
                    
        
       


