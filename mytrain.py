import h5py
import numpy
import os
import argparse
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,CSVLogger
import mymodels
from myutils import psnr, ssim


def read_data(file):
    file = h5py.File(str(data_file),'r')
    inputs =file['data'][:].astype(numpy.float32)/255.0        
    label = file['label'][:].astype(numpy.float32)/255.0
    #check:
    print 'data shape:',inputs.shape
    return inputs, label


def train():
    
    #set model
    model =mymodels.DenseNet(42,42,3)
    model = model.create_model()
    optimizer = Adam(lr=args.lr)
    model.compile(optimizer=optimizer,loss='mean_squared_error', metrics=[psnr, ssim])
    #model.load_weights(os.path.join(root_dir,'weights.h5'))
    print model.summary()

    #set dataset
    inputs, label = read_data(data_file)
    #val_data, val_label = read_data(test_data_file)

    # Set up callbacks
    callbacks = []
    callbacks += [ModelCheckpoint(os.path.join(root_dir,'MyWeights','{epoch:02d}-{val_loss:.4f}.h5'),save_weights_only=True,period=1)]
    #callbacks += [ModelCheckpoint('{epoch:02d}-{val_loss:.4f}.h5', monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=True, mode='min')]
    callbacks += [CSVLogger(history_file, append=True)]

    #running
    model.fit(inputs, label,epochs=args.epochs,batch_size=args.batchsize,shuffle=True,
    #validation_data=(test_inputs,test_label)
    validation_split=0.1, 
    callbacks=callbacks)

    #save model
    #model.save_weights('weights.h5')


# Training settings
parser = argparse.ArgumentParser(description='deblocking')
parser.add_argument('--batchsize', type=int, default=128, help='training batch size')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.001')
#parser.add_argument('--data_file', type=str, default='Data10.h5', help='data file')
#parser.add_argument('--history_file', type=str, default='MyHistory.csv', help='record the history')

args = parser.parse_args()

print(args)

root_dir=os.getcwd()
history_file=os.path.join(root_dir,'MyHistory.csv')
data_file=os.path.join(root_dir,'Data10.h5')

if not os.path.exists(os.path.join(root_dir, 'MyWeights')):
    os.mkdir(os.path.join(root_dir, 'MyWeights'))


if __name__=='__main__':
    train()