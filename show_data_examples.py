import lmdb
import base64
from io import BytesIO
from PIL import Image

image_ids = [286314, 141999, 183846]

lmdb_imgs = "datapath/datasets/MUGE/lmdb/valid/imgs"
env_imgs = lmdb.open(lmdb_imgs, readonly=True, create=False, lock=False, readahead=False, meminit=False)
txn_imgs = env_imgs.begin(buffers=True)
for image_id in image_ids:
    image_b64 = txn_imgs.get("{}".format(image_id).encode('utf-8')).tobytes()
    img = Image.open(BytesIO(base64.urlsafe_b64decode(image_b64)))
    img.show()