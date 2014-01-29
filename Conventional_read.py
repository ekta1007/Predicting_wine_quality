import numpy, random
data=[]
with open("D:/Desktop/PROJECT_NAME/wine_combined_binned_2.csv", "r") as source:  
    header=source.readline()
    ncols=header.count(',')+1
    for line in source :
        data.append(line.replace('\n','').split(',')[0:ncols]) # can use [0:header.count(',')+1] to abstact the # of column fields, here [0:13]

# partioning the datset into train & test
numpy.random.shuffle(data) # to enable random sampling later
train_sample_len=int(len(data)*0.7) # 70% train sample
training, test = data[0:train_sample_len], data[train_sample_len:]
train_data=[training[i][0:ncols-1] for i in range(0,train_sample_len)]  
train_label=[training[i][ncols-1] for i in range(0,train_sample_len)]
