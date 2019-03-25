from pylab import *
from PIL import Image

#im = Image.open("/home/moriaty/data/Projects/RetinalUnet/NEW/training/mask/20_training_mask.gif")
#im.show()
imgs=np.empty((565,565,3))
im = Image.open("./NEW/training/images/20_training.tif")
imgs = np.asarray(im)
#imgs = np.transpose(im,(0,3,1,2))
#im2 = (255/(im.max()-im.min())*(im-im.min())).astype(uint8)
print(imgs.dtype,imgs.ndim)
#print(im2.dtype,im2.max())
# Im = Image.fromarray(im2)
# Im.show()
# Im.save("./NEW/training/mask/old20.gif")