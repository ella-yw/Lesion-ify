import pandas as pd, numpy as np, matplotlib.pyplot as plt, keras

from Classification_Report import plot_classification_report
from Confusion_Matrix import plot_confusion_matrix
from History_Plot import plot_history_accslosses

#######################################################

df_train = pd.read_csv("Processed Data/Training/df_train.csv")
df_val = pd.read_csv("Processed Data/Validation/df_val.csv")

num_train_samples = len(df_train); num_val_samples = len(df_val)
train_batch_size = 10; val_batch_size = 10

#######################################################

datagen = keras.preprocessing.image.ImageDataGenerator(preprocessing_function= \
                                                       keras.applications.mobilenet.preprocess_input)

train_batches = datagen.flow_from_directory('Processed Data/Training', target_size=(224, 224), 
                                            batch_size=train_batch_size)
valid_batches = datagen.flow_from_directory('Processed Data/Validation', target_size=(224, 224), 
                                            batch_size=val_batch_size)

#######################################################

mobile = keras.applications.mobilenet.MobileNet()

x = mobile.layers[-6].output
x = keras.layers.Dropout(0.25)(x)
predictions = keras.layers.Dense(7, activation='softmax')(x)
model = keras.models.Model(inputs=mobile.input, outputs=predictions)

for layer in model.layers[:-23]:
    layer.trainable = False

#######################################################

model.compile(keras.optimizers.Adam(lr=0.01), loss='categorical_crossentropy', 
              metrics=['accuracy'])

class_weights = { 
                  0: 1.0, # akiec
                  1: 1.0, # bcc
                  2: 1.0, # bkl
                  3: 1.0, # df
                  4: 1.5, # mel # Trying to make the model more sensitive to Melanoma.
                  5: 0.5, # nv  # Trying to make the model less sensitive to Melanocytic Nevi.
                  6: 1.0, # vasc
                }

checkpoint = keras.callbacks.ModelCheckpoint("Model.hdf5", monitor='val_acc', verbose=1, 
                                             mode='max', save_best_only=True, save_weights_only=False)
#reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2, 
#                                              verbose=1, mode='max', min_lr=0.00001)
csv_logger = keras.callbacks.CSVLogger('History.csv', append=True, separator=',')

history = model.fit_generator(train_batches, class_weight = class_weights,
                              steps_per_epoch = np.ceil(num_train_samples / train_batch_size), 
                              validation_data = valid_batches,
                              validation_steps = np.ceil(num_val_samples / val_batch_size), 
                              epochs = 15, verbose = 1, callbacks = [checkpoint, csv_logger #, reduce_lr
                                                                    ])
#######################################################

model = keras.models.load_model('Model.hdf5')

datagen = keras.preprocessing.image.ImageDataGenerator(preprocessing_function= \
                                                       keras.applications.mobilenet.preprocess_input)
test_batches = datagen.flow_from_directory('Processed Data/Validation', target_size=(224, 224), 
                                           batch_size=1, shuffle=False)

preds = model.predict_generator(test_batches, verbose=1); preds = [ np.argmax(y) for y in preds ]
labels = list(test_batches.classes) # {'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6}

from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer(); lb.fit(labels)
truth = lb.transform(labels); pred = lb.transform(preds)

from sklearn.metrics import roc_auc_score, accuracy_score, cohen_kappa_score, classification_report, confusion_matrix
roc_auc = roc_auc_score(truth, pred)

acc = accuracy_score(labels, preds)

k = cohen_kappa_score(labels, preds)

class_labels = ['Actinic_Keratoses', 'Basal_Cell_Carcinoma', 'Benign_Keratosis', 'Dermatofibroma ', 'Melanoma', 'Melanocytic_Nevi', 'Vascular_Skin_Lesions']

classif_rep = classification_report(labels, preds, target_names=class_labels); cr = classif_rep.split("\n")
plot_classification_report(cr[0] + '\n\n' + cr[2] + '\n' + cr[3] + '\n' + cr[4] + '\n' + cr[5] 
                           + '\n' + cr[6] + '\n' + cr[7] + '\n' + cr[8] + '\n\n' + cr[10] + '\n',
                           title = 'Classification Report', cmap='binary')
plt.savefig('Classification_Report.png', dpi=200, format='png', bbox_inches='tight'); plt.close();

cnf_matrix = confusion_matrix(labels, preds)
plot_confusion_matrix(cnf_matrix, classes=class_labels, title='Confusion Matrix', normalize=True, cmap=plt.cm.binary)
plt.savefig('Confusion_Matrix.png', dpi=200, format='png', bbox_inches='tight'); plt.close();

akiec_acc = cnf_matrix[0][0]/sum(cnf_matrix[0]);
bcc_acc = cnf_matrix[1][1]/sum(cnf_matrix[1]);
bkl_acc = cnf_matrix[2][2]/sum(cnf_matrix[2]);
df_acc = cnf_matrix[3][3]/sum(cnf_matrix[3]);
mel_acc = cnf_matrix[4][4]/sum(cnf_matrix[4]);
nv_acc = cnf_matrix[5][5]/sum(cnf_matrix[5]);
vasc_acc = cnf_matrix[6][6]/sum(cnf_matrix[6]);

categorized_acc = "Actinic_Keratoses:      " + str(akiec_acc*100) + "%"
categorized_acc += "\nBasal_Cell_Carcinoma:   " + str(bcc_acc*100) + "%"
categorized_acc += "\nBenign_Keratosis:       " + str(bkl_acc*100) + "%"
categorized_acc += "\nDermatofibroma:         " + str(df_acc*100) + "%"
categorized_acc += "\nMelanoma:               " + str(mel_acc*100) + "%"
categorized_acc += "\nMelanocytic_Nevi:       " + str(nv_acc*100) + "%"
categorized_acc += "\nVascular_Skin_Lesions:  " + str(vasc_acc*100) + "%"

f = open("Metrics.txt", "w")
f.write("\n::::::::::::::::: OUTPUT DERIVATIONS :::::::::::::::::" +

        '\n\n-----------------' +
        
        '\n\nAccuracy: ' + str(acc*100) + '%' +
        '\nCohen\'s Kappa Co-efficient (K): ' + str(k) +
        '\nArea Under the Receiver Operating Characteristics (AUROC): ' + str(roc_auc) +
        
        '\n\n-----------------' +
        
        '\n\nCategorized Accuracies: \n\n' + str(categorized_acc) +
        
        '\n\n-----------------' +
        
        '\n\nClassification Report: \n\n' + str(classif_rep))
f.close()

plot_history_accslosses()
plt.savefig("History_Plot.png", dpi=500, format='png', bbox_inches='tight'); plt.close(); 

#######################################################