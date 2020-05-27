from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import fish_finder
import numpy as np
import pandas as pd
import class_reader
import os.path
from keras.optimizers import SGD
from os import path
import keras as keras
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from tensorflow.keras import initializers
from numpy.random import seed

def create_dataset(filenames, threshold, num_days, clean=True):

    output_file = "data/training_data/fish_3days_" + str(threshold) +  "db.csv"
    if path.exists(output_file):
        return output_file

    # Finds all the cfish simultaneously
    cfish = fish_finder.cfish_df() #TODO: get cfish for all the 3 days
    cfish.columns = ["Fish"]
    threshold_value, replace_value = 1, 1
    cfish["Fish"] = cfish["Fish"].where(cfish["Fish"] <= threshold_value, replace_value)
    for day in range(num_days):

        sfish = class_reader.get_sfish(filenames[day])


        # Since EK80 is faulty we need to remove obvisouly erroneous data
        sfish_clean = []
        if clean:
            for i in range(len(sfish)-1):
                if max(sfish[i].compensatedFrequencyResponse) >= 0 or sfish[i].Depth > 14:
                    continue
                elif min(sfish[i].compensatedFrequencyResponse) <= -100 or sfish[i].Depth < 0:
                    continue
                else:
                    sfish_clean.append(sfish[i])
        else:
            sfish_clean = sfish

        data = {"Depth": [x.Depth for x in sfish_clean],
                "AlongshipAngle": [x.AlongshipAngle for x in sfish_clean],
                "AthwartshipAngle": [x.AthwartshipAngle for x in sfish_clean],
                "sa": [x.sa for x in sfish_clean],
                "Frequency": [(x.compensatedFrequencyResponse) for x in sfish_clean]}
        sfish_df = pd.DataFrame(data, columns=["Depth", "AlongshipAngle", "AthwartshipAngle", "sa", "Frequency"], index=[str(x.time) for x in sfish_clean])

        sfish_df = sfish_df.sort_index()

        start = "2019-03-0" + str(3+day) + " 08:00:00"
        end   = "2019-03-0" + str(3+day) + " 17:00:00"

        _,_,common = fish_finder.get_common(sfish_df, cfish,start, end)

        common.to_csv(output_file, mode="a", header=False)

    return output_file



def get_dataset(filename):
    dataset = pd.read_csv(filename, delimiter=",", header=None)
    time = dataset[0]
    depth = dataset[1]
    alongshipangle = dataset[2]
    athwartshipangle = dataset[3]
    sa = dataset[4]
    freq = dataset[5].values
    #x = [t for t in x]
    for i, row in enumerate(freq):
        freq[i] = row[1:-1]
        freq[i] = list(map(float, freq[i].split(",")))

    freq = np.array([i for i in freq])
    freq = freq.transpose()

    x = np.array([depth, alongshipangle, athwartshipangle, sa])
    x = np.append(x, freq, axis=0)

    y = dataset[6].values
    return x.transpose(), y


#def normalizer#(data):
    """
    Normalize per feature
    """
   # normalized_data = np.empty(data.shape)
   # for col in data.T:
   #     mean = np.mean(col)
   #     std = np.std(col)

   #     z = [(x - mean)/std for x in col]
   #     np.append(normalized_data, z)
   # return normalized_data

  #  normalize




def create_NN():
    # define the keras model
    model = Sequential()
    model.add(Dense(1004, input_dim=1004, activation='relu'))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))
    return model

def garbage(model, x, y):
    seed(1) # this is a numpy seed. Keras relies on numpy for seeding weights so this makes the weights the same every time
    x = normalize(x, axis=0, norm="l2") #axis 0 is per column, axis 1 is per row, l2 is euclidean norm
    y = np.array(y).reshape(-1,1)


    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.3, shuffle=True)


    # about y
    # 1 - fish
    # 0 - not fish

    encoder = OneHotEncoder()
    encoder.fit(y)
    y_train = encoder.transform(y_train).toarray()
    y_val = encoder.transform(y_val).toarray()

    #y = [-1 if z == 0 else 1 for z in y]


    # compile the keras model
    opt = SGD(lr=0.1, momentum=0.9)
    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(thresholds=0.5, class_id=1), tf.keras.metrics.Recall(name="Recall", thresholds=0.5, class_id=1)])

    # fit the keras model on the dataset
    model.fit(x_train, y_train, epochs=200, batch_size=1000, validation_data=(x_val,y_val), shuffle=True)

    # evaluate the keras model
    loss, accuracy,  precision, recall = model.evaluate(x_val, y_val)
    print('Accuracy: %.3f, Precision: %.3f, Recall: %.3f' % (accuracy*100, precision*100, recall*100))

    predictions_train = model.predict_classes(x_train)
    predictions_val   = model.predict_classes(x_val)

    print("===========TRAINING METRICS========")
    report_train = classification_report(encoder.inverse_transform(y_train),predictions_train)
    print(report_train)
    con_mat_train = tf.math.confusion_matrix(encoder.inverse_transform(y_train), predictions_train)
    print(con_mat_train.numpy())
    print("===========VALIDATION METRICS========")
    report_val = classification_report(encoder.inverse_transform(y_val),predictions_val)
    print(report_val)
    con_mat_val = tf.math.confusion_matrix(encoder.inverse_transform(y_val), predictions_val)
    print(con_mat_val.numpy())




    return "hi"


def main():
    threshold = 100
    filenames = ["data/sonar_processed/fish_singletargets_0303_" + str(threshold) + "db_slow.bin",
                 "data/sonar_processed/fish_singletargets_0304_" + str(threshold) + "db_slow.bin",
                 "data/sonar_processed/fish_singletargets_0305_" + str(threshold) + "db_slow.bin"]
    dataset_file = create_dataset(filenames, threshold, num_days=3, clean=True)
    x, y = get_dataset(dataset_file)

    model = create_NN()

    garbage(model, x, y)






if __name__=="__main__":
    main()