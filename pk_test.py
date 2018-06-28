import os
import csv
import random
import datetime
from glob import glob

import numpy as np
from scipy.io import wavfile

os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf



categories = ['yes','no','on','off','left','right','up','down','go','stop', 'unknown', 'silence']
id2name = { i:x for i,x in enumerate(categories)}
name2id = { id2name[i]:i for i in id2name }

data_root = './data/train'
traindata_csvfile = 'train.csv'
validdata_csvfile = 'valid.csv'
slience_files =  glob(data_root+'/audio/{}/*.wav'.format('_background_noise_'))


epoch=20
batchsize=64
categories_cnt = len(categories)


print(id2name)
#print(name2id)
#print('\n'.join(aa), len(aa))
#(1, 469, 128, 1)

trainset = []
validset = []
with open(traindata_csvfile,'r') as traincsv:
    csvreader = csv.reader(traincsv)
    trainset = [[x[0], int(x[1])] for x in csvreader]
    #trainset = [[wavfile.read(x[0])[1], int(x[1])] for x in csvreader]
    trainfilename , trainlabel_ori = zip(*trainset)
    trainfilename , trainlabel_ori = list(trainfilename), list(trainlabel_ori)

with open(validdata_csvfile,'r') as validcsv:
    csvreader = csv.reader(validcsv)
    validset = [[x[0], int(x[1])] for x in csvreader]
    #validset = [[wavfile.read(x[0])[1], int(x[1])] for x in csvreader]
    validfilename , validlabel_ori = zip(*validset)
    validfilename , validlabel_ori = list(validfilename), list(validlabel_ori)
traincnt = len(trainlabel_ori)
testcnt  = len(validlabel_ori)
print(trainfilename[:3])
print(trainlabel_ori[:3])
#print(trainset[:3])

def hz_to_mel(freq):
    return 1127. * tf.log(1.0 + (freq / 700.))

def mel_to_hz(mel):
    return 700.*(tf.exp(mel/1127.)-1.)

def multi_ffts_to_mel(freq_array, n_mels=128):
    melfreq_array = tf.expand_dims(hz_to_mel(freq_array),0)
  
    mel_edges = tf.lin_space(hz_to_mel(tf.reduce_min(freq_array)), #or just use 0
                           hz_to_mel(tf.reduce_max(freq_array)), #or SR/2
                           n_mels+2)
  
    lower_edge_mel, center_mel, upper_edge_mel =tf.split(tf.contrib.signal.frame(mel_edges, 3, 1, axis=-1), 3, axis=-1)

    wt_down = (melfreq_array - lower_edge_mel) / (center_mel - lower_edge_mel)
    wt_up = (upper_edge_mel - melfreq_array) / (upper_edge_mel - center_mel)
  
    mel_weights_matrix = tf.maximum(0.0, tf.minimum(wt_down, wt_up))
    center_mel_freqs = mel_to_hz(center_mel) 
  
    return mel_weights_matrix, center_mel_freqs

def audioframes2logmelspec(b_framed_signal, n_ffts=5, 
                           wvls_per_window_hinge=16, n_mel=128, 
                           fft_l1=1024, sr=16000):
    
    fft1_space = tf.lin_space(0., .5, 1+fft_l1//2)[1:]
    freq_list =[sr*fft1_space] 
    n_wv_list =[fft_l1*fft1_space]

    fft_list =[tf.spectral.rfft(b_framed_signal)[:,:,1:]]
  
    for i in range(1,n_ffts):
        fft_lnew = fft_l1//2**i
        fftnew_space = tf.lin_space(0., .5, 1+fft_lnew//2)[1:]
    
        freq_list.append(sr*fftnew_space)
        n_wv_list.append(fft_lnew*fftnew_space)
    
        frames_new = b_framed_signal[:, :, (fft_l1-fft_lnew)//2:(fft_l1-fft_lnew)//2+fft_lnew]
        fft_list.append(tf.spectral.rfft(frames_new)[:,:,1:])
    
  
    freq_concat = tf.concat(freq_list, axis=-1)
    n_wv_concat = tf.concat(n_wv_list, axis=-1)
    fft_concat = tf.concat(fft_list, axis=-1)
    
    magnitude_spectros = tf.abs(fft_concat)

    mel_wts, center_mel_freqs = multi_ffts_to_mel(freq_concat, n_mel)
    wvls_wts = tf.where(n_wv_concat>wvls_per_window_hinge, wvls_per_window_hinge/n_wv_concat, tf.ones_like(n_wv_concat))
  
    mel_spectro=tf.tensordot(magnitude_spectros, (mel_wts*tf.expand_dims(wvls_wts,0)),axes = [[2], [1]])

    log_mel_spectro = tf.log(mel_spectro+1e-7)
  
    #return tf.expand_dims(log_mel_spectro, -1), center_mel_freqs
    return tf.expand_dims(tf.squeeze(log_mel_spectro), -1), center_mel_freqs

def preprocessing(filepath, label):
    sr, wav = wavfile.read(filepath)
    signal = wav.astype(np.float32) / np.iinfo(np.int16).max
    print(type(signal))
    label = np.eye(categories_cnt,dtype=np.float32)[label]
    return sr, signal, label
    #return filepath, label
    #label = np.eye(categories_cnt,dtype=np.float32)[label]
    #def preprocessing(signal, label):
    #    signal = tf.cast(signal, dtype=tf.float32)
    #audio_binary = tf.read_file(filepath)
def preprocessing2(sr,signal,label):
    #label = tf.one_hot(label, categories_cnt,dtype=tf.float32)
    b_signals = tf.expand_dims(signal, axis=0)

    b_framed_signal = tf.contrib.signal.frame(b_signals, 
                                      frame_length=1024, 
                                      frame_step = 32)
    log_mel_spectro, center_mel_freqs = audioframes2logmelspec(b_framed_signal, sr=sr)
    log_mel_spectro = tf.reshape(log_mel_spectro, [469, 128, 1])
    label = tf.reshape(label, [categories_cnt])
    #return log_mel_spectro, label
    return signal,label

def get_iterator(data_a, data_b):
    print(type(data_a[0]), type(data_b[0]))
    inputs = tf.data.Dataset.from_tensor_slices((data_a, data_b))
    ##preprocessing = lambda a,b: tf.py_func(preprocessing, [a,b], [ tf.float32, tf.float32, tf.int32])
    preprocessing = lambda a,b: tf.py_func(preprocessing, [a,b], [tf.float32,  tf.float32, tf.float32 ])
    ##.map(preprocessing2,num_parallel_calls=4)
    
    inputs = inputs.map(preprocessing).map(preprocessing2).shuffle(buffer_size=batchsize*3).prefetch(buffer_size=batchsize*6).batch(batchsize).repeat()
    iterator = inputs.make_initializable_iterator()
    iterator_init_op = iterator.initializer
    return iterator, iterator_init_op

def net(layer, reuse , is_training):
    layer = tf.reshape(layer, [-1, 469, 128, 1])
    """    layer, endpoint = nets.resnet_v1.resnet_v1_50(layer,5, is_training=is_training, reuse=reuse)
    layer = tf.layers.flatten(layer)
    out = tf.nn.softmax(layer) if not is_training else layer
    return out
    """
    with tf.variable_scope('net', reuse = reuse,):
        
        layer = tf.layers.conv2d(layer, 32, kernel_size=3, padding='same', trainable=is_training)
        layer = tf.layers.max_pooling2d(layer, pool_size=2, strides=2)

        layer = tf.layers.conv2d(layer, 64, kernel_size=3, padding='same', trainable=is_training)
        layer = tf.layers.max_pooling2d(layer, pool_size=2, strides=2)
        
        layer = tf.layers.conv2d(layer, 128, kernel_size=3, padding='same', trainable=is_training)
        layer = tf.layers.max_pooling2d(layer, pool_size=2, strides=2)
        
        layer = tf.layers.flatten(layer)
        layer = tf.layers.dense(layer,128, trainable=is_training)
        
        layer = tf.layers.dropout(layer, rate=0.5, training=is_training)
        layer = tf.layers.dense(layer,categories_cnt , trainable=is_training)
        out = tf.nn.softmax(layer) if not is_training else layer
        return out


# TF DATA API를 이용한 train iterator 정의
train_iterator, train_iterator_init_op = get_iterator(trainfilename, trainlabel_ori)
trainwav , trainlabel  = train_iterator.get_next()
#logit_train = net(trainwav , reuse = False, is_training=True)

# TF DATA API를 이용한 test iterator 정의
test_iterator ,test_iterator_init_op  = get_iterator(validfilename, validlabel_ori)
testwav , testlabel  = test_iterator.get_next()
#logit_test  = net(testwav  , reuse = True, is_training=False)

# 학습을 위한 코스트 함수 정의
#cost = tf.reduce_mean(tf.losses.softmax_cross_entropy(trainlabel, logit_train))
#optimizer = tf.train.AdamOptimizer().minimize(cost)

# 테스트 단계에서 프리딕션 정확도 측정
#testcorrect_pred = tf.equal(tf.argmax(logit_test,1), tf.argmax(testlabel,1))
#testaccuracy = tf.reduce_mean(tf.cast(testcorrect_pred, tf.float32))


print("#################################")
starttime =  datetime.datetime.now()

step = traincnt//batchsize

config=tf.ConfigProto()
config.gpu_options.allow_growth = True
print(step," step!! ", starttime)
with tf.Session(config = config) as sess:
    #함수와 iterator들을 초기화
    sess.run(tf.global_variables_initializer())
    sess.run(train_iterator_init_op)
    sess.run(test_iterator_init_op)
    
    print("optimize start!!")
    #print(sess.run(trainimg))
    for i in range(1,2):#step*epoch):
        # 정의한 optimizer 연산으로 학습!!
        print(sess.run(trainwav))
        print(sess.run(trainlabel))
        #sess.run(optimizer)
        if i%(step//30)==0:
            print("*", end='')
        if i % step == 0:
            # 입력 배치 사이즈 만큼 정확도 측정을 하기때문에 테스트 데이터셋 사이즈 만큼 루프 후 엔빵
            print("TEST!!! : ", datetime.datetime.now()-starttime)
            #acc = 0
            #for j in range((testcnt//batchsize)//10):
            #    acc += sess.run(testaccuracy)
            #print("step {}, accuracy : {}".format(i,acc/((testcnt//batchsize)+1)))
            # 이게 되나 모르겠는데.. 랜덤하게 argumentation 하려고...
            #sess.run(train_iterator_init_op)
    # 입력 배치 사이즈 만큼 정확도 측정을 하기때문에 valid 데이터셋 사이즈 만큼 루프 후 엔빵
    #validacc = 0
    #for j in range(validcnt//batchsize):
    #    validacc += sess.run(validaccuracy)
    #print("finally, accuracy : {}".format(validacc/((validcnt//batchsize)+1)))
    #acc = 0
    #for j in range(testcnt//batchsize):
    #    acc += sess.run(testaccuracy)
    #print("step {}, accuracy : {}".format(i,acc/((testcnt//batchsize)+1)))

print("================== e n d ==================")
