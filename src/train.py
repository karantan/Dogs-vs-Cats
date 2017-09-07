from src.vgg16 import Vgg16
from src.utils import get_logger

logger = get_logger()

batch_size = 4
epoch = 2
path = 'data/dogscats/'

vgg = Vgg16(file_path='models/', vgg_weights='vgg16.h5')
batches = vgg.get_batches(path + 'train', batch_size=batch_size)
val_batches = vgg.get_batches(path + 'valid', batch_size=batch_size * 2)
vgg.finetune(batches)
vgg.fit(batches, val_batches, epoch=epoch)

model_path = 'models/dogs_cats_epoch{}.h5'.format(epoch)
logger.info('Saving model to {} ...'.format(model_path))
vgg.model.save(model_path)
logger.info('Done.')
