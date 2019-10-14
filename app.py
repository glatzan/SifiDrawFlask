import base64
import skimage

from io import BytesIO
from flask import Flask
from flask import jsonify
from flask import request
from flask_cors import CORS
from PIL import Image
from skimage import img_as_float32
from skimage.morphology import skeletonize
from tf_unet import image_util
from tf_unet import unet
from tf_unet import util
import numpy as np
from PIL import Image
from skimage.io import imsave
from skimage import img_as_uint

from tf_unet import unet

app = Flask(__name__)
CORS(app)


def decode(base64_string):
    if isinstance(base64_string, bytes):
        base64_string = base64_string.decode("utf-8")

    imgdata = base64.b64decode(base64_string)
    img = skimage.io.imread(imgdata, plugin='imageio')
    return img


def encode(image) -> str:
    # convert image to bytes
    with BytesIO() as output_bytes:
        PIL_image = Image.fromarray(skimage.img_as_ubyte(image))
        PIL_image.save(output_bytes, 'JPEG')  # Note JPG is not a vaild type here
        bytes_data = output_bytes.getvalue()

    # encode bytes to base64 string
    base64_str = str(base64.b64encode(bytes_data), 'utf-8')
    return base64_str


def runUnet(data, model, chan, classes, layers, features):
    message = request.get_json(force=True)
    img = decode(data["data"])
    img = np.array(img, np.float32)

    net = unet.Unet(channels=chan, n_class=classes, layers=layers, features_root=features)

    ny = img.shape[0]
    nx = img.shape[1]
    img = img.reshape(1, ny, nx, 1)
    img -= np.amin(img)
    img /= np.amax(img)

    prediction = net.predict(model, img)

    mask = prediction[0, ..., 1] > 0.1

    data["data"] = encode(img_as_uint(mask))

    return data


@app.route('/unetg1', methods=["POST"])
def unet_graft_1():
    message = request.get_json(force=True)
    print("-----")
    print(message["id"])

    img = base64.b64decode(message["data"])
    img2 = skimage.io.imread(img, plugin='imageio')

    skeleton = skeletonize(img_as_float32(img2))

    message["data"] = encode(skeleton)

    print(message)

    return message


@app.route('/unetg3', methods=["POST"])
def unet_graft3():
    message = request.get_json(force=True)
    return runUnet(message, "model/unet_graft/model.ckpt", 1, 2, 3, 24)


@app.route('/unetg4', methods=["POST"])
def unet_graft4():
    message = request.get_json(force=True)
    return runUnet(message, "model/graft_f24_s3000_e20_graft/model.ckpt", 1, 2, 3, 24)


if __name__ == '__main__':
    app.run()
