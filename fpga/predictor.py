import pynq.lib.dma
from pynq import allocate
import numpy as np
from pynq import Overlay
import csv
import time
import warnings
warnings.filterwarnings("ignore")

ol = Overlay('mlp4002.bit')

dma_send = ol.axi_dma_0
dma_recv = ol.axi_dma_0

input_buffer = allocate(shape=(93,), dtype=np.uint32)
output_buffer = allocate(shape=(1,), dtype=np.uint32)


file = open('features_int3.csv')
csvreader = csv.reader(file)
header = next(csvreader)
confusion_matrix = [[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]]
sum = 0 
count = 0
times = []
for row in reversed(list(csvreader)):
    for i in range(len(row)):
        if i == len(row) - 1:
            label = int(row[i], 10)
            count = count + 1
            start = time.time()
            dma_send.sendchannel.transfer(input_buffer)
            dma_recv.recvchannel.transfer(output_buffer)
            dma_send.sendchannel.wait()
            dma_recv.recvchannel.wait()
            end = time.time()
            confusion_matrix[output_buffer[0]][label] = confusion_matrix[output_buffer[0]][label] + 1
            if label == output_buffer[0]:
                sum = sum + 1
            print("prediction accuracy:" + str(sum/count) + " time elapsed: " + str(end-start))
            times.append(end-start)
             
        else:
            input_buffer[i] = int(row[i], 10)



# print(outp)
for row in confusion_matrix:
    print(row)

print("Avg time: " + str(np.mean(times)))
