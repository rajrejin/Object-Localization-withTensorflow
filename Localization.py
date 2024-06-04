import os
import zipfile
import urllib.request
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image, ImageDraw
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPool2D, BatchNormalization, Dropout

# Download the file from the URL
url = "https://github.com/hfg-gmuend/openmoji/releases/latest/download/openmoji-72x72-color.zip"
if not os.path.exists('openmoji-72x72-color.zip'):
    urllib.request.urlretrieve(url, 'openmoji-72x72-color.zip')

# Create a new directory named "emojis"
if not os.path.exists('emojis'):
    os.makedirs('emojis')

# Extract the contents of the zip file into the "emojis" directory
with zipfile.ZipFile('openmoji-72x72-color.zip', 'r') as zip_ref:
    zip_ref.extractall('emojis')

# Define a dictionary of emojis, each represented by a name and a file name
emojis = {
    0: {'name': 'happy', 'file': '1F642.png'},
    1: {'name': 'laughing', 'file': '1F602.png'},
    2: {'name': 'skeptical', 'file': '1F928.png'},
    3: {'name': 'sad', 'file': '1F630.png'},
    4: {'name': 'cool', 'file': '1F60E.png'},
    5: {'name': 'whoa', 'file': '1F62F.png'},
    6: {'name': 'crying', 'file': '1F62D.png'},
    7: {'name': 'puking', 'file': '1F92E.png'},
    8: {'name': 'nervous', 'file': '1F62C.png'}
}

# Enable interactive mode
plt.ion()

# Create a new figure and display each emoji image in a subplot
plt.figure(figsize=(9, 9))
for i, (j, e) in enumerate(emojis.items()):
    plt.subplot(3, 3, i + 1)
    plt.imshow(plt.imread(os.path.join('emojis', e['file'])))
    plt.xlabel(e['name'])
    plt.xticks([])
    plt.yticks([])
plt.show()
plt.pause(5)  # Display the plot for 5 seconds
plt.close()  # Close the plot

# Open each emoji image, add a white background, and store the new image in the emojis dictionary
for class_id, values in emojis.items():
    png_file = Image.open(os.path.join('emojis', values['file'])).convert('RGBA')
    png_file.load()
    new_file = Image.new("RGB", png_file.size, (255, 255, 255))
    new_file.paste(png_file, mask=png_file.split()[3])
    emojis[class_id]['image'] = new_file

# Define a function to create an example
def create_example():
    class_id = np.random.randint(0,9)
    image = np.ones((144, 144, 3)) * 255
    row = np.random.randint(0, 72)
    col = np.random.randint(0, 72)
    image[row: row + 72, col: col + 72, :] = np.array(emojis[class_id]['image'])
    return image.astype('uint8'), class_id, (row + 10) / 144, (col + 10)/ 144

# Use the function to create an example
image, class_id, row, col = create_example()
plt.imshow(image)
plt.show()
plt.pause(5)
plt.close()

#Plotting Bounding Boxes
def plot_bounding_box(image, gt_coords, pred_coords = [], norm = False):
    if norm:
        image *= 255
        image = image.astype('uint8')
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)

    row, col = gt_coords
    row *= 144
    col*= 144
    draw.rectangle((col, row, col + 52, row + 52), outline = 'green', width = 3)
    if len(pred_coords) == 2:
        row, col = pred_coords
        row *= 144
        col*= 144
        draw.rectangle((col, row, col + 52, row + 52), outline = 'red', width = 3)
    return image

# Define a data generator that generates training examples
def data_generator(batch_size = 16):
    while True:
        x_batch = np.zeros((batch_size, 144, 144, 3))
        y_batch = np.zeros((batch_size, 9))
        bbox_batch = np.zeros((batch_size, 2))

        for i in range(0, batch_size):
            image, class_id, row, col = create_example()
            x_batch[i] = image/255
            y_batch[i, class_id] = 1.0
            bbox_batch[i] = np.array([row, col])
        yield {'image': x_batch}, {'class_out': y_batch, 'box_out': bbox_batch}

#Use the generator to create an example
example, label = next(data_generator(1))
image = example['image'][0]
class_id = np.argmax(label['class_out'][0])
coords = label['box_out'][0]

#Visualize the generated example
image = plot_bounding_box(image, coords, norm = True)
plt.imshow(image)
plt.title(emojis[class_id]['name'])
plt.show()
plt.pause(5)
plt.close()

#Build the model
input_ = Input(shape = (144, 144, 3), name = 'image')

x = input_

for i in range (0, 5):
    n_filters = 2**(4 + i)
    x = Conv2D(n_filters, 3, activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(2)(x)

x = Flatten()(x)
x = Dense(256, activation = 'relu')(x)

class_out = Dense(9, activation = 'softmax', name = 'class_out')(x)
box_out = Dense(2, name = 'box_out')(x)

model = tf.keras.models.Model(input_, [class_out, box_out])
model.summary()

#Define a custom Intersection Over Union (IOU) metric
class IoU(tf.keras.metrics.Metric):
    def __init__(self, **kwargs):
        super(IoU, self).__init__(**kwargs)
        self.iou = self.add_weight(name='iou', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        def get_box(y):
            rows, cols = y[:, 0], y[:, 1]
            rows, cols = rows * 144, cols * 144
            y1, y2 = rows, rows + 52
            x1, x2 = cols, cols + 52
            return x1, y1, x2, y2

        def get_area(x1, y1, x2, y2):
            return tf.math.abs(x2 - x1) * tf.math.abs(y2 - y1)

        gt_x1, gt_y1, gt_x2, gt_y2 = get_box(y_true)
        p_x1, p_y1, p_x2, p_y2 = get_box(y_pred)

        # Ensure p_x1, p_y1, p_x2, and p_y2 are of the same type as gt_x1, gt_y1, gt_x2, and gt_y2
        p_x1 = tf.cast(p_x1, tf.float64)
        p_y1 = tf.cast(p_y1, tf.float64)
        p_x2 = tf.cast(p_x2, tf.float64)
        p_y2 = tf.cast(p_y2, tf.float64)

        i_x1 = tf.maximum(gt_x1, p_x1)
        i_y1 = tf.maximum(gt_y1, p_y1)
        i_x2 = tf.minimum(gt_x2, p_x2)
        i_y2 = tf.minimum(gt_y2, p_y2)

        i_area = get_area(i_x1, i_y1, i_x2, i_y2)
        u_area = get_area(gt_x1, gt_y1, gt_x2, gt_y2) + get_area(p_x1, p_y1, p_x2, p_y2) - i_area

        iou = tf.math.divide(i_area, u_area)
        self.iou.assign(tf.reduce_mean(iou))


    def result(self):
        return self.iou

    def reset_states(self):
        self.iou.assign(0.)

# Compile the model
model.compile(
    loss = {
        'class_out': 'categorical_crossentropy',
        'box_out': 'mse'
    },
    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3),
    metrics = {
        'class_out': 'accuracy',
        'box_out': IoU(name = 'iou')
    }
)

# Testing the model
def test_model(model, test_datagen, epoch):
    plt.figure(figsize=(6, 6))  # Create a new figure for each epoch
    classes = random.sample(list(emojis.keys()), 4)  # Randomly select 4 classes
    for i in range(len(classes)):
        example, label = next(test_datagen)
        x = example['image']
        y = label['class_out']
        box = label['box_out']

        pred_y, pred_box = model.predict(x)

        pred_coords =  pred_box[0]
        gt_coords = box[0]
        pred_class = np.argmax(pred_y[0])
        image = x[0]

        gt =  emojis[np.argmax(y[0])]['name']
        pred_class_name = emojis[pred_class]['name']

        image = plot_bounding_box(image, gt_coords, pred_coords, norm =True)
        color = 'green' if gt == pred_class_name else 'red'

        plt.subplot(2, 2, i + 1)  # Create a subplot for each emoji
        plt.imshow(image)
        plt.xlabel(f'Pred: {pred_class_name}', color = color)
        plt.ylabel(f'GT: {gt}', color = color)
        plt.xticks([])
        plt.yticks([])
    plt.suptitle(f'Epoch: {epoch + 1}')  # Add epoch number to the title of the figure
    plt.show()

def test(model, epoch):
    test_datagen = data_generator(1)

    # Call test_model with epoch number
    test_model(model, test_datagen, epoch)

class ShowTestImages(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = None):
        test(self.model, epoch)

def lr_schedule(epoch, lr):
    if (epoch + 1) % 5 == 0:
        lr *= 0.2
    return max(lr, 3e-7)

model.fit(
    data_generator(),
    epochs=2,
    steps_per_epoch=500,
    callbacks=[
        ShowTestImages(),
        tf.keras.callbacks.EarlyStopping(monitor='box_out_iou', patience=3, mode='max'),
        tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    ]
)