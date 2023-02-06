import pickle
import sys
import numpy as np
import traceback
from PIL import Image
import segmentation_models_pytorch as smp
import cv2

import torch 

ACTIVE = bytes([1])
NOT_ACTIVE = bytes([0])

# While debigging this script, you may add a convenient full path to the log file name to find it easily.
# Otherwise, find the log file in your Slicer binaries installation folder.
# Here is an example how to print debug info in the log file:
# f.write("resizeInputArray, input_array shape: {}\n".format(input_array.shape))

f = open(r"LivePredictionsLog.txt", "a")

def resizeInputArray(input_array):
  """
  Resize the image and add it to the current buffer
  :param input_image: np.array
  :return: np.array resized and rescaled for AI model
  """
  input_image = Image.fromarray(input_array)  # PIL.Image knows np.array reverses axes: M -> image.width, F -> image.height
  resized_input_array = np.array(  # np.array will be (F, M) again, but resize happens in (M, F) axis order
      input_image.resize(( 256, 256,))
    )

  resized_input_array = cv2.cvtColor(resized_input_array, cv2.COLOR_BGR2RGB) 
      
  #Normalize
  resized_input_array = resized_input_array.astype(np.float32)
  mean, std = resized_input_array.mean(), resized_input_array.std()
  resized_input_array = (resized_input_array - mean) / std
  resized_input_array = np.clip(resized_input_array, -1.0, 1.0)
  resized_input_array = (resized_input_array + 1.0) / 2.0

  #Transform numpy array to tensor
  resized_input_array = np.transpose(resized_input_array, (2, 0, 1))
  resized_input_array = torch.from_numpy(resized_input_array).to(DEVICE).unsqueeze(0)

  return resized_input_array

def resizeOutputArray(y):
  output_array = y.squeeze().cpu().numpy().round()
  output_array = output_array.astype(np.uint8)*255
  upscaled_output_array = cv2.resize(output_array, (input_array.shape[2], input_array.shape[1]))

  return upscaled_output_array


# Do first read at the same time that we instantiate the model, this ensures that the TF graph is written/ready right away when we start live predictions.

try:
  input = sys.stdin.buffer.read(1)  # Read control byte
  if input == ACTIVE:
    input_length = sys.stdin.buffer.read(8) # Read data length
    input_length = int.from_bytes(input_length, byteorder='big')
    input_data = sys.stdin.buffer.read(input_length)  # Read the data
  elif input == NOT_ACTIVE:
    f.close()
    sys.exit()

  input_data = pickle.loads(input_data)  # Axes: (F, M) because slicer.util.array reverses axis order from IJK to KJI

  # Load AI model
  # If you are doing Batch Size == 1 predictions, this tends to speed things up (wrapping .call in a @tf.function decorator)

  self.unet_model = smp.Unet(
    encoder_name='resnet18', 
    encoder_weights=None, 
    classes=1, 
    in_channels=3,
    decoder_use_batchnorm=True,
    activation='sigmoid',
    )
  self.unet_model.load_state_dict(torch.load(input_data['model_path']))
  self.unet_model.to(DEVICE)
  # Run a dummy prediction to initialize TF
  """
  input_array = resizeInputArray(input_data['volume'])
  _ = model(input_array, training=False)

  # Starting loop to receive images until status set to NOT_ACTIVE

  while True:
    input = sys.stdin.buffer.read(1)  # Read control byte
    # f.write("Active status: {}\n".format(str(input)))

    if input == ACTIVE:
      input_length = sys.stdin.buffer.read(8) # Read data length
      input_length = int.from_bytes(input_length, byteorder='big')
      input_data = sys.stdin.buffer.read(input_length) # Read the data
    else:
      break

    input_data = pickle.loads(input_data)
    input_array = input_data['volume']
    input_transform = input_data['transform']
    if input_array.shape[0] == 1:
      input_array = input_array[0, :, :]
    input_array = resizeInputArray(input_array)

    y = model(input_array, training=False)
    # f.write("Prediction shape in while loop: {}\n".format(y.shape))
    output_array = resizeOutputArray(y)

    output = {}
    output['prediction'] = output_array
    output['transform'] = np.array(input_transform)  # Make a copy of the input transform, because it has already lapsed in Slicer
    pickled_output = pickle.dumps(output)

    sys.stdout.buffer.write(pickled_output)
    sys.stdout.buffer.flush()

  f.write("Exiting by request from Slicer\n")
  f.close()
  sys.exit()
  """

except:
  f.write("ERROR in process script:\n{}\n".format(traceback.format_exc()))
  f.close()
  sys.exit()
