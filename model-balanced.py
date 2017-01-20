from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam 
from keras.models import model_from_json
from itertools import zip_longest
import cv2
import numpy as np
import csv, argparse
import os, errno
import matplotlib.pyplot as plt

# Function to compute number of lines in a file
# Arguments:
#  fname - the filename of the file to analyse
# Returns:
#  length of file
#
def file_len(fname):
    i = -1
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

# Function to remove a file if it exists
# Arguments:
#  filename - the filename to remove
#
def remove(filename):
    try:
        os.remove(filename)
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise

            
# Function to load driving logs into in memory data structures
# Arguments:
#  logs - a list of one or more csv format log file names
#  path - the full path to the log files
# Returns:
#  A tuple (f, y)
#  f - list of full path filenames to individual images
#  y - list of associated steering angle for each image
# Notes:
#  The function extracts the filenames of the left, right and 
#  center images it does not load the images themselves.  The function
#  also makes an adjustment of +/- 0.2 on the center steering angle 
#  for the left and right images respectively.  Only returns image 
#  filenames and steering angles if all three images exist as files.
#
def load_driving_logs(logs, path):
    f = []
    y = []
    for d in logs:
        log = d+'driving_log.csv'
        with open(log,'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                imgc_file = path + d + row[0].strip()
                imgl_file = path + d + row[1].strip()
                imgr_file = path + d + row[2].strip()
                if (os.path.isfile(imgc_file) & os.path.isfile(imgl_file) & os.path.isfile(imgr_file)):
                    # Center image - angle as is
                    f.append(imgc_file)
                    y.append(np.float32(row[3]))
                    # Left image - center angle add 0.2
                    f.append(imgl_file)
                    y.append(np.float32(row[3])+0.2)
                    # Right image - center angle subtract 0.2
                    f.append(imgr_file)
                    y.append(np.float32(row[3])-0.2)
    return f, y
  
    
# Function to randomly split driving logs in to train and
# validation subsets.
# Arguments:
#  f - the list of image filenames
#  y - the list of associated steering angles
#  train_percent - the fraction in range 0.0 : 1.0 of data for training
#  seed - random number seed to use - defaults to 1973
# Returns:
#  tuple (ft, yt, fv, yv)
#  ft - the list of training image filenames
#  yt - the list of training steering angles
#  fv - the list of validation image filenames
#  yv - the list of validation steering angles
#  
def split_driving_log(f, y, train_percent, seed=1973):
    ft = []
    yt = []
    fv = []
    yv = []
    np.random.seed(seed)
    for idx, img in enumerate(f):
        if (np.random.random() <= train_percent):
            ft.append(img)
            yt.append(y[idx])
        else:
            fv.append(img)
            yv.append(y[idx])
    return ft, yt, fv, yv


# Function to load an image from file to numpy array
# Arguments
#  f - filename of an RGB image
# Returns
#  img - a numpy array representing the image
#
def load_image(f):
    img = cv2.imread(f,-1)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img 


# Function to apply a gamma correction to an image
# Arguments
#  img - the image as a numpy array
#  correction - the adjustment factor
# Returns
#  A numpy array of same shape as img but with brightness adjusted
# Notes
#  Corrections larger than 1 make the image lighter
#  while corrections less than 1 make image darker
#
def gamma_correction(img, correction):
    img = img/255.0
    img = cv2.pow(img, correction)
    return np.uint8(img*255)


# Function to preprocess images, in this case trimming
#  the image by 60 pixels at the top and 20 pixels at the
#  bottom and then resizing to 64x64 pixles
# Arguments
#  img - input image to preprocess
# Returns
#  A preprocessed image as a numpy array
#
def preprocess_image(img):
    img = img[60:140,:,:]
    img = cv2.resize(img,(64, 64))
    return(img)
    

# Given a list of steering angle bin cut points, bin the 
#  training data into buckets
# Arguments
#  f - a list of image file names
#  y - the corresponding steering angles
#  bins - a list of bin cut-point lower bounds
# Returns
#  A tuple (fb, yb)
#  fb - a dictionary of lists of image filename values keyed on bucket index names
#  yb - a dictionary of lists of steering angle values keyed on bucket index names
#
def bin_data(f, y, bins):
    fb = {}
    yb = {}
    for idx, img in enumerate(f):
        bin = str((i for i,v in enumerate(bins) if v >= y[idx]).__next__())
        if bin in fb:
            fb[bin].append(img)
            yb[bin].append(y[idx])
        else:
            fb[bin] = [img]
            yb[bin] = [y[idx]]
    return fb, yb
 

# Generator to serve up a batch of stratified randomly sampled images 
# across the previously binnned steering angle buckets - binned using bin_data()
# Arguments
#  fb - list of image filenames
#  yb - list of corresponding steering angles
#  n - the required batch size
# Returns
#  Yields a tuple (x, y) 
#  x - list of stratified sampled image file names
#  y - list of corresponding steering angles for the stratified sample of images
# 
def generate_balanced(fb, yb, n):
    while True:
        xs = []
        ys = []
        for _ in range(n):
            # First choose a bucket with equal probabilty
            bin = np.random.randint(low=1, high=8)
            l = len(fb[str(bin)])
            # Now sample image from that bucket
            i = np.random.randint(low=0,high=l)
            img, angle = generate_image(load_image(fb[str(bin)][i]),yb[str(bin)][i])
            xs.append(img)
            ys.append(angle)
        yield (np.asarray(xs), np.asarray(ys))
        

# Function to augment images by applying the folllowing transformations:
#  1) cropping top 60 and bottom 20 pixels from image
#  2) resizing images to (64, 64, 3)
#  3) random brightness gamma correction (between 0.2 and 4)
#  4) random flip with probability 0.5
#  5) random X and Y transalation of upto 10 pixels
# Arguments
#  img - original image of shape (160, 320, 3) 
#  y - 
# Returns
#  tuple (img, angle)
#  img- augmented image of shape (64, 64, 3)
#  angle - augmented image adjusted angle to allow for x translation
#
def generate_image(img, y):
    
    angle = y

    X_OFFSET_RANGE = 10
    Y_OFFSET_RANGE = 10
    X_OFFSET_ANGLE = 0.2
    
    img = preprocess_image(img)
    
    bright_factor = 0.2 + (3.8 * np.random.uniform())
    img = gamma_correction(img, bright_factor)

    if (np.random.uniform() > 1.0):
        img = np.fliplr(img)
        angle = -1.0 * angle

    x_translation = (X_OFFSET_RANGE * np.random.uniform()) - (X_OFFSET_RANGE / 2)
    y_translation = (Y_OFFSET_RANGE * np.random.uniform()) - (Y_OFFSET_RANGE / 2)

    angle = angle + ((x_translation / X_OFFSET_RANGE) * 2) * X_OFFSET_ANGLE
    t = np.float32([[1, 0, x_translation], [0, 1, y_translation]])
    img = cv2.warpAffine(img, t, (img.shape[1], img.shape[0]))

    return (img, angle)


# Function to create Kera's deep convolutional network
# Parameters
#  None
# Returns
#  Keras model with specific architecture
#
def get_model():
    ch, row, col = 3, 64, 64  

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
            input_shape=( row, col, ch),
            output_shape=( row, col, ch)))
    model.add(Convolution2D(32, 3, 3, border_mode="same"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, border_mode="same"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))
    model.add(ELU())
    model.add(Convolution2D(128, 3, 3, border_mode="same"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))   
    model.add(ELU())
    model.add(Flatten())
    model.add(Dense(512))
    model.add(ELU())
    model.add(Dense(64))
    model.add(ELU())
    model.add(Dense(16))
    model.add(ELU())
    model.add(Dense(1))
    return model


# These are the driving logs I want to use to train the model
logs = [ 'data/two-laps-middle-forward/', 
         'data/two-laps-middle-backwards/', 
         'data/two-laps-recovery-forward/']

# Load and bin the data into buckets based on steering angle
f, y = load_driving_logs(logs, '/home/mattwg/Projects/carnd-cloning-experiments/')
fb, yb = bin_data(f, y, bins = (-999, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 999 ))

print([ 'bin:{}={}'.format(k,len(fb[k])) for k in sorted(fb.keys())])
print(sum([len(fb[k]) for k in fb.keys()]))

# Check distributions:
g = generate_balanced(fb, yb, 10000)
fg, yg = g.__next__()
plt.hist(yg, bins = 20)

# Parse command line arguments 
parser = argparse.ArgumentParser(description='Train Model')
parser.add_argument('--model', type=str, help='Optional path to model to continue training')
args = parser.parse_args()

# If model specified load from disk, else create a new model
if args.model:
    print('Loading model for continued training!')
    with open(args.model, 'r') as f:
        json = f.read()
        model = model_from_json(json)
else:
    print('Training new model!')
    model = get_model()

n_train = len(f)
print(n_train)

# Training parameters
n_epochs = 10
batch_size = 50
n_batches = 10 

model.compile(optimizer=Adam(lr=0.00001), loss="mse")

if args.model:
    model.load_weights('model.h5')
    
# Train model
model.fit_generator(
    generate_balanced(fb, yb, batch_size),
    samples_per_epoch=n_batches,
    nb_epoch=n_epochs, verbose=1)

# Serialize model and weights
json = model.to_json()
with open('model.json', 'w') as f:
    f.write(json)  
model.save_weights('model.h5')



