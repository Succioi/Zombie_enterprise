import pandas as pd
from keras.models import load_model


model = load_model('train/weight.h5')
data_test = pd.read_csv('data_data.csv')
data_test = data_test.loc[:,['ent_type','registered_fund']]

