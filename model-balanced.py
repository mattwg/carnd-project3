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


def file_len(fname):
    i = -1
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def remove(filename):
    try:
        os.remove(filename)
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise

            
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
                    f.append(imgc_file)
                    y.append(np.float32(row[3]))
                    f.append(imgl_file)
                    y.append(np.float32(row[3])+0.2)
                    f.append(imgr_file)
                    y.append(np.float32(row[3])-0.2)
    return f, y


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



def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def load_image(f):
    img = cv2.imread(f,-1)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img 


def extract_csv(log):
    f = []
    y = []
    with open(log,'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            f.append(row[0].strip())
            y.append(np.float32(row[3]))
            f.append(row[1].strip())
            y.append(np.float32(row[3])+0.2)
            f.append(row[2].strip())
            y.append(np.float32(row[3])-0.2)
    return f, y


def gamma_correction(img, correction):
    img = img/255.0
    img = cv2.pow(img, correction)
    return np.uint8(img*255)


def preprocess_image(img):
    img = img[60:140,:,:]
    img = cv2.resize(img,(64, 64))
    return(img)
    
    
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
 
    
def generate_balanced(fb, yb, n):
    while True:
        xs = []
        ys = []
        for _ in range(n):
            bin = np.random.randint(low=1, high=8)
            l = len(fb[str(bin)])
            i = np.random.randint(low=0,high=l)
            img, angle = generate_image(load_image(fb[str(bin)][i]),yb[str(bin)][i])
            xs.append(img)
            ys.append(angle)
        yield (np.asarray(xs), np.asarray(ys))
        
        
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

            
def generator_all_batch(n, img_dir, logfile):
    f, y = extract_csv(logfile)
    l = len(f)
    while True:
        xs = []
        ys = []
        for i in range(0,l,n):
            xs.append(preprocess_image(load_image(img_dir+f[i])))
            ys.append(np.float32(y[i]))
            yield (np.asarray(xs), np.asarray(ys))

        
def get_model():
    ch, row, col = 3, 64, 64  

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
            input_shape=( row, col, ch),
            output_shape=( row, col, ch)))
    model.add(Convolution2D(3, 1, 1, border_mode="same"))
    model.add(Convolution2D(32, 3, 3, border_mode="same"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, border_mode="same"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(ELU())
    model.add(Convolution2D(128, 3, 3, border_mode="same"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(ELU())
    model.add(Flatten())
    model.add(Dense(512))
    model.add(ELU())
    model.add(Dropout(0.5))
    model.add(Dense(64))
    model.add(ELU())
    model.add(Dropout(0.5))
    model.add(Dense(16))
    model.add(ELU())
    model.add(Dropout(0.5))
    model.add(Dense(1))
    return model


#####################################################################
# Model training here!
#####################################################################

# Define data folders
logs = [ 'data/two-laps-middle-forward/', 
         'data/two-laps-middle-backwards/', 
         'data/two-laps-recovery-forward/']

# Load files names and steering angles
f, y = load_driving_logs(logs, '/home/mattwg/Projects/carnd-cloning-experiments/')

# Bin data
fb, yb = bin_data(f, y, bins = (-999, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 999 ))


# Print out bin sizes
print([ 'bin:{}={}'.format(k,len(fb[k])) for k in sorted(fb.keys())])
print(sum([len(fb[k]) for k in fb.keys()]))

# Parse arguments - check for optional model parameter:
#   if provided continue training, 
#   else start from random weights

parser = argparse.ArgumentParser(description='Train Model')
parser.add_argument('--model', type=str, help='Optional path to model to continue training')
args = parser.parse_args()

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

# Define batch size for stochastic gradient descent
n_epochs = 50
batch_size = 400
n_batches = 10000 

model.compile(optimizer=Adam(lr=0.00001), loss="mse")

if args.model:
    model.load_weights('model.h5')
    
model.fit_generator(
    generate_balanced(fb, yb, batch_size),
    samples_per_epoch=n_batches,
    nb_epoch=n_epochs, verbose=1)

# Serialize model
json = model.to_json()
with open('model.json', 'w') as f:
    f.write(json)  
model.save_weights('model.h5')



