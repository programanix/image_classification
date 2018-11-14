# -*- coding: utf-8 -*-
import os
from flask import Flask, render_template
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
## TF
import numpy as np
import json
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from IPython.display import display
from PIL import Image
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.applications import ResNet50
import json
##
my_model = None
image_size = 224

app = Flask(__name__)
app.config['SECRET_KEY'] = 'my super secret key'
app.config['UPLOADED_PHOTOS_DEST'] = '/tmp' 

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)  # set maximum file size, default is 16MB

class UploadForm(FlaskForm):
    photo = FileField(validators=[FileAllowed(photos, u'Image only!'), FileRequired(u'File was empty!')])
    submit = SubmitField(u'Upload')


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    form = UploadForm()
    pred = [['',''] for i in range(3)]
    if form.validate_on_submit():
        filename = photos.save(form.photo.data)
        file_url = photos.url(filename)
        volume = '/tmp/'
        pred = get_prediction(volume+filename)
        pred = [ [i[1],i[2]] for i in pred]
        return render_template('submit.html', form=form, file_url=file_url,tf_pred1=pred[0][0],tf_pred2=pred[1][0],tf_pred3=pred[2][0],tf_pred_p1=pred[0][1],tf_pred_p2=pred[1][1],tf_pred_p3=pred[2][1] )
    else:
        file_url = None
    
    return render_template('index.html', form=form, file_url=file_url)
    

######TF part

def read_and_prep_images(img_paths, img_height=image_size, img_width=image_size):
    r= ""
    logf = open("/tmp/image.log", "w")
    try:
      imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]
      img_array = np.array([img_to_array(img) for img in imgs])
      r = preprocess_input(img_array)
    except Exception as e:     # most generic exception you can catch
      logf.write("{0}: {1}\n".format(str("Exception"), str(e)))
      logf.close()
     
    return r

def decode_predictions(preds, top=5, class_list_path=None):
  """Decodes the prediction of an ImageNet model.
  Arguments:
      preds: Numpy tensor encoding a batch of predictions.
      top: integer, how many top-guesses to return.
      class_list_path: Path to the canonical imagenet_class_index.json file
  Returns:
      A list of lists of top class prediction tuples
      `(class_name, class_description, score)`.
      One list of tuples per sample in batch input.
  Raises:
      ValueError: in case of invalid shape of the `pred` array
          (must be 2D).
  """
  if len(preds.shape) != 2 or preds.shape[1] != 1000:
    raise ValueError('`decode_predictions` expects '
                     'a batch of predictions '
                     '(i.e. a 2D array of shape (samples, 1000)). '
                     'Found array with shape: ' + str(preds.shape))
  CLASS_INDEX = json.load(open(class_list_path))
  results = []
  for pred in preds:
    top_indices = pred.argsort()[-top:][::-1]
    result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
    result.sort(key=lambda x: x[2], reverse=True)
    results.append(result)
  return results

def get_prediction(filename):
   global my_model
   if my_model is None:
         my_model = ResNet50(weights='weights/resnet50_weights_tf_dim_ordering_tf_kernels.h5')
   test_data = read_and_prep_images([filename])
   preds = my_model.predict(test_data)
   most_likely_labels = decode_predictions(preds, top=3, class_list_path='weights/imagenet_class_index.json')
   return most_likely_labels[0]

if __name__ == '__main__':
    app.run(host='0.0.0.0')
