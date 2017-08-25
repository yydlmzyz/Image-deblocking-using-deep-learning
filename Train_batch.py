import MyModel
import h5py
import numpy
from pathlib import Path

#setting parameters
img_width = 42
img_height = 42
img_channel = 3
train_steps=50000
batch_size=128

#setting path and file
save_dir = Path('.')
data_file=save_dir/'TrainDataHigh.h5'
weights_dir = save_dir / 'MyWeights'
if weights_dir.exists():
    print 'folder esxited'
else:
    weights_dir.mkdir()

deblocking_model=MyModel.create_model(img_height,img_width,img_channel)
deblocking_model.load_weights('/home/lhj/wjq/deblock_keras/WeightsBSD10.h5')


file = h5py.File(str(data_file),'r')

for i in range(train_steps):
    ran=numpy.sort(numpy.random.choice(file['data'].shape[0],batch_size,replace=False))
    batch_data=file['data'][ran,:,:,:].astype(numpy.float32)/255.0
    batch_label=file['label'][ran,:,:,:].astype(numpy.float32)/255.0
    #predict=deblocking_model.predict(batch_data)
    #origin_loss=deblocking_model.evaluate(batch_data, predict, batch_size=batch_size)
    loss=deblocking_model.train_on_batch(batch_data,batch_label)
    if i%400==0 and i>0:
        print 'i:',i ,loss
        deblocking_model.save_weights('/home/lhj/wjq/deblock_keras/MyWeights/weights%d.h5' % (i))







