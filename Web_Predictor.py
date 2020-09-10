def pred(link, keras, pd, os):
    
    keras.backend.clear_session()
    
    img = keras.preprocessing.image.ImageDataGenerator(rescale=1./255).flow_from_dataframe(dataframe=pd.DataFrame([[link, "0"]], columns=['filename', 'pred']), 
                                                          directory='img_to_predict/', x_col='filename', y_col='pred', target_size=(224, 224), color_mode='rgb')
    model = keras.models.load_model("../Model.hdf5")
    prediction = model.predict_generator(img, verbose=1)[0]
    
    conf_dict = {
                  prediction[0]: "Actinic Keratoses (Intraepithelial Carcinoma/Bowen's Disease)", 
                  prediction[1]: "Basal Cell Carcinoma/Cancer", 
                  prediction[2]: "Benign Keratosis (Solar Lentigines/Seborrheic Keratoses)", 
                  prediction[3]: "Dermatofibroma (Cutaneous Fibrous Histiocytoma)", 
                  prediction[4]: "Malignant Melanoma", 
                  prediction[5]: "Melanocytic Nevi (Nevocytic Nevus)", 
                  prediction[6]: "Vascular Lesion (Lymphatic Malformation)"
                }
    wiki_dict = {
                  "Actinic Keratoses (Intraepithelial Carcinoma/Bowen's Disease)": 'https://en.wikipedia.org/wiki/Actinic_keratosis',
                  "Basal Cell Carcinoma/Cancer": 'https://en.wikipedia.org/wiki/Basal-cell_carcinoma',
                  "Benign Keratosis (Solar Lentigines/Seborrheic Keratoses)": 'https://en.wikipedia.org/wiki/Seborrheic_keratosis',
                  "Dermatofibroma (Cutaneous Fibrous Histiocytoma)": 'https://en.wikipedia.org/wiki/Benign_fibrous_histiocytoma',
                  "Malignant Melanoma": 'https://en.wikipedia.org/wiki/Melanoma',
                  "Melanocytic Nevi (Nevocytic Nevus)": 'https://en.wikipedia.org/wiki/Melanocytic_nevus',
                  "Vascular Lesion (Lymphatic Malformation)": 'https://en.wikipedia.org/wiki/Vascular_anomaly',
                }
        
    confs = sorted([ key for key in sorted(conf_dict.keys()) ], reverse=True)
    classes = [ conf_dict[conf] for conf in confs ]
    
    msg_main = '<a href="' + wiki_dict[classes[0]] + '" style="text-decoration:none;color:white;"><b>' + classes[0] + '</b></a><br/>'
    msg_main += '(' + str(confs[0]*100) + '% Precision)'; msg_all = ""
    for i in range(len(classes)): 
        try: msg_all += "<b>" + classes[i+1] + "</b>: " + str(round(confs[i+1]*100, 3)) + "% Conf.<br/>"
        except IndexError: pass
    
    path = 'img_to_predict/' + link
    
    os.remove(path); print("File Removed!")
    
    return msg_main, msg_all