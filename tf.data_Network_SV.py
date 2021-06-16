from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow as tf

BATCH_SIZE = 64
EPOCHS = 50

print("[INFO] loading training data...")
((trainX, trainLabels), (textX, testLabels)) = cifar10.load_data()

trainAug = Sequential([
	preprocessing.Rescaling(scale=1.0 / 255),
	preprocessing.RandomFlip("horizontal_and_vertical"),
	preprocessing.RandomZoom(
		height_factor=(-0.05, -0.15),
		width_factor=(-0.05, -0.15)),
	preprocessing.RandomRotation(0.3)
])

testAug = Sequential([
	preprocessing.Rescaling(scale=1.0 / 255)
])

trainDS = tf.data.Dataset.from_tensor_slices((trainX, trainLabels))
trainDS = (trainDS
	.shuffle(BATCH_SIZE * 100)
	.batch(BATCH_SIZE)
	.map(lambda x, y: (trainAug(x), y),
		 num_parallel_calls=tf.data.AUTOTUNE)
	.prefetch(tf.data.AUTOTUNE)
)

testDS = tf.data.Dataset.from_tensor_slices((textX, testLabels))
testDS = (testDS
	.batch(BATCH_SIZE)
	.map(lambda x, y: (testAug(x), y),
		num_parallel_calls=tf.data.AUTOTUNE)
	.prefetch(tf.data.AUTOTUNE)
)

print("[INFO] initializing model...")
model = Sequential()
model.add(Conv2D(32, (3, 3), padding="same",
	input_shape=(32, 32, 3)))
model.add(Activation("relu"))

model.add(Conv2D(64, (3, 3), padding="same",
	input_shape=(32, 32, 3)))
model.add(Activation("relu"))
model.add(Conv2D(64, (3, 3), padding="same",
	input_shape=(32, 32, 3)))
model.add(Activation("relu"))

model.add(Conv2D(128, (3, 3), padding="same",
	input_shape=(32, 32, 3)))
model.add(Activation("relu"))
model.add(Conv2D(128, (3, 3), padding="same",
	input_shape=(32, 32, 3)))
model.add(Activation("relu"))
model.add(Conv2D(128, (3, 3), padding="same",
	input_shape=(32, 32, 3)))
model.add(Activation("relu"))

model.add(Dense(256))

model.add(Flatten())
model.add(Dense(10))
model.add(Activation("softmax"))

print("[INFO] compiling model...")
model.compile(loss="sparse_categorical_crossentropy",
	optimizer="sgd", metrics=["accuracy"])

print("[INFO] training model...")
H = model.fit(
	trainDS,
	validation_data=testDS,
	epochs=EPOCHS)
	
(loss, accuracy) = model.evaluate(testDS)
print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))
