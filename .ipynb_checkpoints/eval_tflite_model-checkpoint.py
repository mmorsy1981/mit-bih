import numpy as np
import pandas as pd

#import tensorflow as tf

import tflite_runtime.interpreter as tflite

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import OneHotEncoder

#Read and arrange data for tensorflow
import opendatasets as od

od.download("https://www.kaggle.com/mohammedfarag/mitbihcsv")

data_set_path=os.path.join("mitbihcsv","mit_bih_data_set.csv")

data_set = pd.read_csv(data_set_path,dtype=float)

num_classes=6

data_set_np = data_set.to_numpy()

X = data_set_np[:,:-1]
y = data_set_np[:,-1]

scalar = StandardScaler()
X = scalar.fit_transform(X)

X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.2,stratify=y, random_state=42)

# One hot Encoding of y
enc = OneHotEncoder()

enc.fit(y_trn.reshape(-1, 1))
y_trn_oh = enc.transform(y_trn.reshape(-1, 1)).toarray()

enc.fit(y_tst.reshape(-1, 1))
y_tst_oh = enc.transform(y_tst.reshape(-1, 1)).toarray()

X_trn_tf = np.expand_dims(X_trn, axis=2)
X_tst_tf = np.expand_dims(X_tst, axis=2)

X_tst = X_tst.astype('float32')
X_tst_tf = X_tst_tf.astype('float32')


tflite_model_file = 'mit_bih.tflite'

# Load TFLite model and allocate tensors.
with open(tflite_model_file, 'rb') as fid:
    tflite_model = fid.read()
    
interpreter = tflite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

#input_index = interpreter.get_input_details()[0]["index"]
#output_index = interpreter.get_output_details()[0]["index"]

# Load TFLite model and allocate tensors.
#interpreter = tf.lite.Interpreter(model_content=float_tflite_model)
#interpreter.allocate_tensors()

#get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

#Preprocess the image to required size and cast
input_shape = input_details[0]['shape']
input_data = np.expand_dims(X_tst_tf[0], 0)

#set the tensor to point to the input data to be inferred
input_index = interpreter.get_input_details()[0]["index"]
interpreter.set_tensor(input_details[0]['index'], input_data)

#Run the inference
interpreter.invoke()
output_details = interpreter.get_tensor(output_details[0]['index'])

# A helper function to evaluate the TF Lite model using "test" dataset.
def evaluate_tflite_model(tflite_model):
  # Initialize TFLite interpreter using the model.
  interpreter = tf.lite.Interpreter(model_content=tflite_model)
  interpreter.allocate_tensors()
  input_tensor_index = interpreter.get_input_details()[0]["index"]
  output = interpreter.tensor(interpreter.get_output_details()[0]["index"])

  # Run predictions on every image in the "test" dataset.
  prediction_outputs = []
  for data in X_tst_tf:
    # Pre-processing: add batch dimension and convert to float32 to match with
    # the model's input data format.
    test_data = np.expand_dims(data, axis=0).astype(np.float32)
    interpreter.set_tensor(input_tensor_index, test_data)

    # Run inference.
    interpreter.invoke()

    # Post-processing: remove batch dimension and find the digit with highest
    # probability.
    prediction = np.argmax(output()[0])
    prediction_outputs.append(prediction)

  # Compare prediction results with ground truth labels to calculate accuracy.
  accurate_count = 0
  for index in range(len(prediction_outputs)):
    if prediction_outputs[index] == y_tst[index]:
      accurate_count += 1
  accuracy = accurate_count * 1.0 / len(prediction_outputs)

  return accuracy

# Evaluate the TF Lite float model. You'll find that its accurary is identical
# to the original TF (Keras) model because they are essentially the same model
# stored in different format.
tflite_accuracy = evaluate_tflite_model(tflite_model)
print('tflite model accuracy = %.4f' % tflite_accuracy)
