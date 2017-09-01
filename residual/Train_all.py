import L8
import h5py
import numpy
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau,CSVLogger
from pathlib import Path

#setting parameters
img_width = 42
img_height = 42
img_channel = 3

#setting path and file
save_dir = Path('.')
history_file = save_dir / 'MyHistory.csv'
data_file=save_dir/'TrainDataBSDr.h5'
weights_dir = save_dir / 'MyWeights'
if weights_dir.exists():
    print 'folder esxited'
else:
    weights_dir.mkdir()


deblocking_model=L8.create_model(img_height,img_width,img_channel)
#check:
print deblocking_model.summary()

# Set up callbacks
callbacks = []
callbacks += [ModelCheckpoint(str(weights_dir/'{epoch:02d}-{val_loss:.6f}.h5'),save_weights_only=True,period=1)]
callbacks += [CSVLogger(str(history_file), append=True)]#save history for visualization
callbacks += [EarlyStopping(monitor='val_loss', patience=5)]
callbacks += [ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=6, epsilon=0.0001, cooldown=1, min_lr=0)]#reduce learing rate if necessary

#1st step:load h5 data
file = h5py.File(str(data_file),'r')

train_data =file['data'][:].astype(numpy.float32)/255.0      
train_label = file['label'][:].astype(numpy.float32)
print train_label

#check:
print 'train_data.shape:',train_data.shape,'train_label.shape:',train_label.shape ,'train_data.itemsize:',train_data.itemsize

#3rd step:fit network


deblocking_model.load_weights('/home/lhj/wjq/deblock_keras/weights1.h5')


deblocking_model.fit(train_data,train_label,
epochs=40,
batch_size=128,
shuffle=True,
validation_split=0.1, 
callbacks=callbacks)

