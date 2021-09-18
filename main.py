import numpy as np
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from sqlalchemy import Table, Column, Integer, String, MetaData, ForeignKey
from sqlalchemy import Float, LargeBinary
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import datetime
from numpy import split
from numpy import array
from math import sqrt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import ConvLSTM2D
import pickle
import time

# define parameters
verbose, epochs, batch_size = 0, 50, 32
kol_neuron = 100
n_layer = 1
ModelPath = "model.h5"
scaler = MinMaxScaler(feature_range=(-1, 1))

# evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(actual, predicted):
	scores = list()
	# calculate an RMSE score for each day
	for i in range(actual.shape[1]):
		# calculate mse
		mse = mean_squared_error(actual[:, i], predicted[:, i])
		# calculate rmse
		rmse = sqrt(mse)
		# store
		scores.append(rmse)
	# calculate overall RMSE
	s = 0
	for row in range(actual.shape[0]):
		for col in range(actual.shape[1]):
			s += (actual[row, col] - predicted[row, col])**2
	score = sqrt(s / (actual.shape[0] * actual.shape[1]))
	# plot forecasts vs observations
	#for j in range(predicted.shape[1]):
	#	show_plot(actual[:, j], predicted[:, j], j + 1)
	return score, scores

# split a univariate dataset into train/test sets
def split_dataset(data):
	# split into standard weeks
	train, test = data[0:735], data[735:854]
	# restructure into windows of weekly data
	train = array(split(train, len(train)/7))
	test = array(split(test, len(test)/7))
	return train, test

# summarize scores
def summarize_scores(name, score, scores):
	s_scores = ', '.join(['%.1f' % s for s in scores])
	print('%s: [%.3f] %s' % (name, score, s_scores))

# convert history into inputs and outputs
def to_supervised(train, n_input, n_out=7):
	# flatten data
	data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
	X, y = list(), list()
	in_start = 0
	# step over the entire history one time step at a time
	for _ in range(len(data)):
		# define the end of the input sequence
		in_end = in_start + n_input
		out_end = in_end + n_out
		# ensure we have enough data for this instance
		if out_end <= len(data):
			x_input = data[in_start:in_end, 0]
			x_input = x_input.reshape((len(x_input), 1))
			X.append(x_input)
			y.append(data[in_end:out_end, 0])
		# move along one time step
		in_start += 1
	return array(X), array(y)

# train the model
def build_model(train, n_steps, n_length, n_input):
	# prepare data
	train_x, train_y = to_supervised(train, n_input)
	n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
	# reshape into subsequences [samples, time steps, rows, cols, channels]
	train_x = train_x.reshape((train_x.shape[0], n_steps, 1, n_length, n_features))
	# reshape output into [samples, timesteps, features]
	train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
	# define model
	model = Sequential()
	model.add(ConvLSTM2D(filters=64, kernel_size=(1, 3), activation='relu', input_shape=(n_steps, 1, n_length, n_features)))
	model.add(Flatten())
	model.add(RepeatVector(n_outputs))
	model.add(LSTM(kol_neuron, activation='relu', return_sequences=True))
	if n_layer == 2:
		model.add(LSTM(kol_neuron, activation='relu', return_sequences=True))
	model.add(TimeDistributed(Dense(200, activation='relu')))
	model.add(TimeDistributed(Dense(1)))
	model.compile(loss='mse', optimizer='adam')
	# fit network
	model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
	return model

# evaluate a single model
def evaluate_model(train, test, n_steps, n_length, n_input):
	# fit model
	model = build_model(train, n_steps, n_length, n_input)
	# history is a list of weekly data
	history = [x for x in train]
	# walk-forward validation over each week
	predictions = list()
	for i in range(len(test)):
		# predict the week
		yhat_sequence = forecast(model, history, n_steps, n_length, n_input)
		# store the predictions
		predictions.append(yhat_sequence)
		# get real observation and add to history for predicting the next week
		history.append(test[i, :])
	# evaluate predictions days for each week
	predictions = array(predictions)
	score, scores = evaluate_forecasts(test[:, :, 0], predictions)
	model.save(ModelPath)
	return score, scores

def show_plot(true, pred, title):
    fig = pyplot.subplots()
    pyplot.plot(true, label='Y_original')
    pyplot.plot(pred, dashes=[4, 3], label='Y_predicted')
    pyplot.xlabel('N_samples', fontsize=12)
    pyplot.ylabel('Instance_value', fontsize=12)
    pyplot.title(title, fontsize=12)
    pyplot.grid(True)
    pyplot.legend(loc='upper right')
    pyplot.show()

# make a forecast
def forecast(model, history, n_steps, n_length, n_input):
	# flatten data
	data = array(history)
	#print("data2 - ", data[0])
	data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
	#rint("data3 - ", data[0])
	# retrieve last observations for input data
	input_x = data[-n_input:, 0]
	#print("input_x1 - ", input_x[0])
	# reshape into [samples, time steps, rows, cols, channels]
	input_x = input_x.reshape((1, n_steps, 1, n_length, 1))
	#print("input_x2 - ", input_x[0])
	# forecast the next week
	yhat = model.predict(input_x, verbose=2)
	# we only want the vector forecast
	yhat = yhat[0]
	return yhat

#Save result to DB
def save_result(table_name, model_path, articul, calc_time, epoch, neuron, n_layer, batch_size, n_steps, n_length, rmse_all, rmse):
	f = open(model_path, 'rb')
	object_model = f.read()
	alchemyEngine = create_engine('postgresql+psycopg2://prognoz:prognoz@10.200.25.18/prognoz', pool_recycle=3600)
	dbConnection = alchemyEngine.connect()
	metadata_obj = MetaData()
	result_model = Table(table_name, metadata_obj,
						 Column('articul', Integer),
						 Column('calc_time', Integer),
						 Column('epoch', Integer),
						 Column('neuron', Integer),
						 Column('n_layer', Integer),
						 Column('batch_size', Integer),
						 Column('n_steps', Integer),
						 Column('n_length', Integer),
						 Column('rmse_all', Float),
						 Column('rmse_mon', Float),
						 Column('rmse_tue', Float),
						 Column('rmse_wed', Float),
						 Column('rmse_thr', Float),
						 Column('rmse_fri', Float),
						 Column('rmse_sut', Float),
						 Column('rmse_sun', Float),
						 Column('model', LargeBinary)
						 )
	ins = result_model.insert().values(articul=articul, calc_time=calc_time, epoch=epoch, neuron=neuron, n_layer=n_layer,
									   batch_size=batch_size, n_steps=n_steps, n_length=n_length, rmse_all=rmse_all,
									   rmse_mon=rmse[0], rmse_tue=rmse[1], rmse_wed=rmse[2], rmse_thr=rmse[3],
									   rmse_fri=rmse[4], rmse_sut=rmse[5], rmse_sun=rmse[6], model=object_model)
	result = dbConnection.execute(ins)
	dbConnection.close()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #load data from postgres
	# Load data
	alchemyEngine = create_engine('postgresql+psycopg2://prognoz:prognoz@10.200.25.18/prognoz', pool_recycle=3600)
	dbConnection = alchemyEngine.connect()
	sql_data = '''SELECT val FROM train_data1 WHERE DATE_PART('year', train_data1.dt) in (2019,2020,2021) order by train_data1.dt'''
	df_data = pd.read_sql(sql_data, dbConnection)
	dbConnection.close()
	names_data = df_data.columns
	d = scaler.fit_transform(df_data)
	data_norm = pd.DataFrame(d, columns=names_data)
	# End load data
	df_train, df_test = split_dataset(df_data)
	#df_train, df_test = split_dataset(data_norm)
	# define the number of subsequences and the length of subsequences
	n_steps, n_length = 1, 7
	table_name = 'result_model'
	#table_name = 'result_model_norm'
	#print("df_train - ", df_train)
	#print("df_test - ", df_test)
	#n_input = n_length * n_steps
	#score, scores = evaluate_model(df_train, df_test, n_steps, n_length, n_input)
	#summarize_scores('lstm', score, scores)
	#interval = time.time() - start_time
	# plot scores
	#days = ['mon', 'tue', 'wed', 'thr', 'fri', 'sat', 'sun']
	#fig = pyplot.subplots()
	#pyplot.plot(days, scores, marker='o', label='lstm')
	#pyplot.show()
	for n_steps in [1, 2]:
		for n_length in [7, 14]:
			for epochs in [25, 50]:
				for kol_neuron in [100, 200]:
					for n_layer in [1, 2]:
						print(f"n_steps:{n_steps}  n_length:{n_length}  epochs:{epochs}  kol_neuron:{kol_neuron}  n_layer:{n_layer}")
						n_input = n_length * n_steps
						start_time = time.time()
						score, scores = evaluate_model(df_train, df_test, n_steps, n_length, n_input)
						# summarize scores
						summarize_scores('lstm', score, scores)
						interval = time.time() - start_time
						# plot scores
						#days = ['mon', 'tue', 'wed', 'thr', 'fri', 'sat', 'sun']
						#fig = pyplot.subplots()
						#pyplot.plot(days, scores, marker='o', label='lstm')
						#pyplot.show()
						save_result(table_name, ModelPath, 1, interval, epochs, kol_neuron, n_layer, batch_size, n_steps, n_length, score, scores)