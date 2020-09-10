import os, random, pandas, keras, pyttsx3, speech_recognition as sr
from twilio.rest import Client; import yagmail;

def speech_to_text():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening..."); audio = r.listen(source) ;
        try: print("Transcribing: ", end=''); inp = r.recognize_google(audio, language = 'en-CA'); print(inp); #en-US #hi-IN 
        except sr.UnknownValueError:
            inp = "Google Speech Recognition could not understand audio"
        except sr.RequestError as e:
            inp = "Could not request results from Google Speech Recognition service; {0}".format(e)
        return inp
    
def comm(confs, classes, path):
    txt = "Would you like me to email you or text you these results, or shall I do both?"; print(txt); engine.say(txt); engine.runAndWait();
    inp = speech_to_text()
    
    if any(substring in inp.lower() for substring in ["no", "nope", "don't", "nothing", "stop"]) == False:
        
        if any(substring in inp.lower() for substring in ["both"]):
            txt = "For sure sir - emailing and texting you the results at om000developer@gmail.com, and to 9024013278, respectively!"; print(txt); engine.say(txt); engine.runAndWait();
            email(confs, classes, path); message(confs, classes)
            txt = "Done!"; print(txt); engine.say(txt); engine.runAndWait();
        elif any(substring in inp.lower() for substring in ["email", "mail"]):
            txt = "Yes sir - sending the results to your gmail!"; print(txt); engine.say(txt); engine.runAndWait();
            email(confs, classes, path)
            txt = "Done!"; print(txt); engine.say(txt); engine.runAndWait();
        elif any(substring in inp.lower() for substring in ["message", "text"]):
            txt = "Alright sir, sending the results to your phone!"; print(txt); engine.say(txt); engine.runAndWait();
            message(confs, classes)
            txt = "Done!"; print(txt); engine.say(txt); engine.runAndWait();
        else:
            txt = "Sorry sir! Couldn't understand your input! Please try again!"; print(txt); engine.say(txt); engine.runAndWait(); comm();
            
    else: txt = "Okay sir - no problem!"; print(txt); engine.say(txt); engine.runAndWait();

       
def email(confs, classes, path):
    
    msg = "Greetings from the Lesion-ify!"
    msg += "\n\nThe Skin Lesion You Captured Is Most Likely An Instance Of <b>" + classes[0] + "</b> (Model Is <b>" + str(confs[0]*100) + "% Confident</b>)."
    msg += "\n\nHowever, Here Is A <span style='text-decoration:underline'>Probabilistic Confidence Breakdown Of Other Possible Conditions</span>:\n"
    
    for i in range(len(classes)): 
        try: msg += "\n" + classes[i+1] + ": " + str(round(confs[i+1]*100, 3)) + "% Conf."
        except IndexError: pass
    
    receiver = "om000developer@gmail.com"
    subject = "Recent Skin Lesion Prediction from Lesion-ify"
    yag = yagmail.SMTP("om000developer@gmail.com", "rashmiom")
    yag.send(to = receiver, subject = subject, contents = [msg, path])
 
def message(confs, classes):
    
    msg = "Greetings from Lesion-ify!"
    msg += "\n\nThe Skin Lesion You Captured Is Most Likely An Instance Of " + classes[0] + " (Model Is " + str(confs[0]*100) + "% Confident)."
    msg += "\n\nHowever, Here Is A Probabilistic Confidence Breakdown Of Other Possible Conditions:\n"
    
    for i in range(len(classes)): 
        try: msg += "\n" + classes[i+1] + ": " + str(round(confs[i+1]*100, 3)) + "% Conf."
        except IndexError: pass
    
    account_sid = 'AC8a21d3810dd09adf8a9fa6873d906803'
    auth_token = '1bdf89c72c0acc0a84a6d3b1e5950aa3'
    client = Client(account_sid, auth_token)
    client.messages \
          .create(body=msg, from_='+19027005652', to='+19024013278')
    
model = keras.models.load_model('Model.hdf5')
datagen = keras.preprocessing.image.ImageDataGenerator(preprocessing_function= \
                                                       keras.applications.mobilenet.preprocess_input)   
engine = pyttsx3.init()
engine.setProperty('rate', 175); 
engine.setProperty('voice', engine.getProperty('voices')[0].id)

intros = ["The network suggests that this is a condition named", "This lesion is most likely an instance of", "Our AI thinks this is probably a case of"]
conditions = ['Actinic Keratoses', 'Basal Cell Carcinoma/Cancer', 'Benign Keratosis', 'Dermatofibroma ', 'Malignant Melanoma', 'Melanocytic Nevi', 'Vascular Lesion']
concs = ["Its confidence in this claim is about X%", "We can say this with approximately X% certainty", "It is X% sure of this"]

while 1:
    
    if len(os.listdir('RealTime')) != 0:
        
        txt = "New Image Detected!"; print(txt); engine.say(txt); engine.runAndWait();
            
        img = datagen.flow_from_dataframe(dataframe=pandas.DataFrame([[os.listdir('RealTime')[0], "0"]], columns=['filename','pred']), 
                                          directory='RealTime/', x_col='filename', y_col='pred', target_size=(224, 224))
        
        prediction = model.predict_generator(img, verbose=1)[0]
        print("Prediction: " + conditions[list(prediction).index(max(prediction))])        
        
        rnd = random.randint(0,2)
        speech = intros[rnd] + " " + conditions[list(prediction).index(max(prediction))] + ". " + concs[rnd].replace("X%", str(round(max(prediction)*100)) + "%")
        
        print("Conveying...", end=''); engine.say(speech); engine.runAndWait(); print("DONE!")
        
        conf_dict = {
                      prediction[0]: "Actinic Keratoses (Intraepithelial Carcinoma/Bowen's Disease)", 
                      prediction[1]: "Basal Cell Carcinoma/Cancer", 
                      prediction[2]: "Benign Keratosis (Solar Lentigines/Seborrheic Keratoses)", 
                      prediction[3]: "Dermatofibroma (Cutaneous Fibrous Histiocytoma)", 
                      prediction[4]: "Malignant Melanoma", 
                      prediction[5]: "Melanocytic Nevi (Nevocytic Nevus)", 
                      prediction[6]: "Vascular Lesion (Lymphatic Malformation)"
                    }
        
        confs = sorted([ key for key in sorted(conf_dict.keys()) ], reverse=True)
        classes = [ conf_dict[conf] for conf in confs ]
        
        path = os.path.join('RealTime', os.listdir('RealTime')[0])
        
        comm(confs, classes, path)

        os.remove(path); print("File Removed!")