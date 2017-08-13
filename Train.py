import MyModel
import h5py
import numpy
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau,CSVLogger
from pathlib import Path

#setting parameters
img_width = 40
img_height = 40
img_channel = 3

#setting path and file
save_dir = Path('.')
history_file = save_dir / 'MyHistory.csv'
data_file=save_dir/'DATA10.h5'
weights_dir = save_dir / 'MyWeights'
if weights_dir.exists():
    print 'folder esxited'
else:
    weights_dir.mkdir()


deblocking_model=MyModel.create_model(img_height,img_width,img_channel)
#check:
print deblocking_model.summary()


# Set up callbacks
callbacks = []
callbacks += [ModelCheckpoint(str(weights_dir/'{epoch:02d}-{val_loss:.6f}.h5'),save_weights_only=True,period=1)]
callbacks += [CSVLogger(str(history_file), append=True)]#save history for visualization
callbacks += [EarlyStopping(monitor='val_loss', patience=10)]
callbacks += [ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=6, epsilon=0.0001, cooldown=1, min_lr=0)]#reduce learing rate if necessary

#1st step:load h5 data
file = h5py.File(str(data_file),'r')

train_data =file['data'][:].astype(numpy.float32)/255.0        
train_label = file['label'][:].astype(numpy.float32)/255.0

#check:
print 'train_data.shape:',train_data.shape,'train_label.shape:',train_label.shape ,'train_data.itemsize:',train_data.itemsize

#3rd step:fit network

'''
#you can load existed weights to train faster:
deblocking_model.load_weights('str(save_dir/'weights_1.h5'))
'''

deblocking_model.fit(train_data,train_label,
epochs=20,
batch_size=128,
shuffle=True,
#validation_data=(test_data,test_label)#
validation_split=0.1, 
callbacks=callbacks)

#4th step:save model
deblocking_model.save('deblocking_model.h5')
deblocking_model.save_weights('deblocking_weights.h5')
