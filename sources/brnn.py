# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# Any results you write to the current directory are saved as output.
import argparse
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import Utils
from keras.preprocessing import text
from keras.callbacks import EarlyStopping, ModelCheckpoint

argparser = argparse.ArgumentParser(
    description='Train and generate submission for Toxic Comment Classification Challenge')

argparser.add_argument('-n', '--rnn',
    required=True,
    help='rnn type LSTM or GRU')

def _main_(args):
  max_features = 20000
  maxlen = 100
  tokenizer = text.Tokenizer(num_words=max_features)
  list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]   
  # Get a model
  model = Utils.get_model(maxlen, max_features, rnn_drop=0.3, rnn=args.rnn)
  batch_size = 512
  epochs = 1
  # Path to write weights file
  file_path="weights_base.best.hdf5"
  # Evaluate criterion
  checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='min')
  # Early stopping criterion
  early = EarlyStopping(monitor="val_acc", mode="min")  
  callbacks_list = [checkpoint, early] # early stopping
  #
  randomize_time = 1
  # Fitting model using randomized learning data (can use generator instead)
  for index in range(randomize_time) :
      X_train, y = Utils.get_train_data(maxlen, max_features, list_classes, tokenizer)
      model.fit(X_train, y, batch_size=batch_size, epochs=epochs, validation_split=0.05, callbacks=callbacks_list)
      
  # Load weights
  model.load_weights(file_path)
  # Get test datas
  X_test = Utils.get_test_data(maxlen, max_features, tokenizer)
  # Prediction on test data
  y_test = model.predict(X_test,verbose=1)
  # Read sample submission
  sample_submission = pd.read_csv("../input/sample_submission.csv")
  # Fill sample submission
  sample_submission[list_classes] = y_test
  # Write sample submission
  sample_submission.to_csv("baseline.csv", index=False)  


if __name__ == '__main__':
    args = argparser.parse_args()    
    if args.rnn != 'LSTM' and  args.rnn != 'GRU':
        argparser.print_help()
    else :
        _main_(args)