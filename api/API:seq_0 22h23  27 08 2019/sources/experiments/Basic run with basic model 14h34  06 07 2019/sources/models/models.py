from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten

def basic_seq_model(input_shape, nb_actions):
	model = Sequential()
	model.add(Flatten(input_shape=(1,) + input_shape))
	model.add(Dense(256))
	model.add(Activation('relu'))
	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(Dense(nb_actions))
	model.add(Activation('linear'))

	return model

def heavy_seq_model(input_shape, nb_actions):
	model = Sequential()
	model.add(Flatten(input_shape=(1,) + input_shape))
	model.add(Dense(256))
	model.add(Activation('tanh'))
	model.add(Dense(256))
	model.add(Activation('tanh'))
	model.add(Dense(256))
	model.add(Activation('tanh'))
	model.add(Dense(128))
	model.add(Activation('tanh'))
	model.add(Dense(128))
	model.add(Activation('tanh'))
	model.add(Dense(128))
	model.add(Activation('tanh'))
	model.add(Dense(nb_actions))
	model.add(Activation('linear'))

	return model

def very_heavy_seq_model(input_shape, nb_actions):
	model = Sequential()
	model.add(Flatten(input_shape=(1,) + input_shape))
	model.add(Dense(256))
	model.add(Activation('tanh'))
	model.add(Dense(256))
	model.add(Activation('tanh'))
	model.add(Dense(256))
	model.add(Activation('tanh'))
	model.add(Dense(128))
	model.add(Activation('tanh'))
	model.add(Dense(128))
	model.add(Activation('tanh'))
	model.add(Dense(128))
	model.add(Activation('tanh'))
	model.add(Dense(64))
	model.add(Activation('tanh'))
	model.add(Dense(64))
	model.add(Activation('tanh'))
	model.add(Dense(64))
	model.add(Activation('tanh'))
	model.add(Dense(nb_actions))
	model.add(Activation('linear'))

	return model