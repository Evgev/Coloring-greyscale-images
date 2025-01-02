from tensorflow.keras.models import Sequential
from tensorflow.keras.models import clone_model
import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb, rgb2gray, xyz2lab
import pprint  

def lab_in_rgb(x, y):
  cur = np.zeros((*x.shape, 3))
  cur[:,:,0] = x
  cur[:,:,1:] = y
  cur_rgb = (lab2rgb(cur) * 255).astype(np.uint8)
  return cur_rgb

def my_plot(history, metrics, show_start_position, round_loss):
  
  count_metrics = len(metrics)

  plt.figure(figsize= (7*count_metrics, 4))
  for n, metric in enumerate(metrics): ## Showing all metrisc into one plot
    position = n + 1
    
    loss = [round(loss, round_loss) for loss in history.history[metric]][show_start_position:]
    val_loss = [round(loss, round_loss) for loss in history.history["val_" + metric]][show_start_position:]
    
    plt.subplot(1, count_metrics ,position)
    plt.plot(loss, color='blue', label = metric) 
    plt.plot(val_loss, color='orange', label = "val_" + metric)
    plt.title(f"{metric} и val_{metric}")
    plt.xlabel('epochs')
    plt.ylabel(metric)
    plt.legend()  
    plt.grid(True)  
        
  plt.show()
  

def cdcoalf(model: Sequential = None
            ,params: dict = None
            ,round_loss: int = 5
            ,data: list = None
            ,metrics: list = None
            ,epochs: int = 10
            ,tensorboard_callback: TensorBoard = None
            ,show_start_position: int = 3):
  
  x_train, y_train, x_val, y_val, x_test, y_test = data
  
  
  
  result_df = list()
  result_df = pd.DataFrame(columns=['optimizer',
                                  'loss_functions',
                                  'evaluate',
                                  'loss',
                                  'val_loss']
                           )
  
  count_metrics = len(metrics)
  
  for i, param in enumerate(params):

    print(param['optimizer'], param['loss_function'])
    new_model = clone_model(model)
    new_model.compile(optimizer = param['optimizer'], loss = param['loss_function'], metrics = metrics)
 
    history = new_model.fit(
      x=np.expand_dims(x_train, axis = 3),
      y=y_train/128,
      # batch_size=10,
      epochs=epochs,
      validation_data = (x_val, y_val/128),
      verbose = 0,
      callbacks=[tensorboard_callback]
    )
    
    
    pprint.pprint(param)
    my_plot(history, metrics, show_start_position, round_loss)
    
    new_row = {
      'optimizer': param['optimizer'],
      'loss_functions':  param['loss_function'],
      'evaluate': new_model.evaluate(x_test, y_test),
      'loss': history.history['loss'],
      'val_loss': history.history['val_loss'],
    }
      
    result_df.loc[len(result_df)] = new_row
    
  
  return result_df, pd.DataFrame(history.history)