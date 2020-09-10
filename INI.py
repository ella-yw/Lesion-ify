import os, time, webbrowser, pyttsx3, speech_recognition as sr

print("""
      
Ask me to do anything regarding Lesion-ify's Platform!

// Launch Web Application
// Launch Phone Interfacer
// Show Me The Pre-Trained Model's Metrics

      """);
      
engine = pyttsx3.init()
engine.setProperty('rate', 175); 
engine.setProperty('voice', engine.getProperty('voices')[0].id)

txt = "Ask me to do anything regarding Lesion-ify's Platform!"; engine.say(txt); engine.runAndWait();
            
r = sr.Recognizer()
    
while 1: 
    
    with sr.Microphone() as source:
        
        print("Listening..."); audio = r.listen(source) ;
        try: print("Transcribing: ", end=''); inp = r.recognize_google(audio, language = 'en-CA'); print(inp); #en-US #hi-IN 
        except sr.UnknownValueError: inp = ""
        except sr.RequestError as e: inp = ""
        
        if any(substring in inp.lower() for substring in ["web", "website", "django"]):
            txt = "Yes sir - launching the web application (please give me a couple of seconds)!"; print(txt); engine.say(txt); engine.runAndWait();
            os.chdir("Web"); os.system("start cmd /K python manage.py runserver"); os.chdir("../"); time.sleep(7.5); webbrowser.open("http://localhost:8000");
            txt = "Done!"; print(txt); engine.say(txt); engine.runAndWait();
            exit()
        elif any(substring in inp.lower() for substring in ["phone", "mobile", "pcloud"]):
            txt = "Yes sir - launching the phone interfacer!"; print(txt); engine.say(txt); engine.runAndWait();
            os.system("start cmd /K python Phone_Predictor.py"); 
            txt = "Done!"; print(txt); engine.say(txt); engine.runAndWait();
            exit()
        elif any(substring in inp.lower() for substring in ["metrics", "results", "outputs", "derivations", "pre-trained", "model"]):
            txt = "Yes sir - opening up metrics of the preetrained model!"; print(txt); engine.say(txt); engine.runAndWait();
            webbrowser.open("file:///P:/Skin Lesion Predictor/Pretrained_Results.html");
            txt = "Done!"; print(txt); engine.say(txt); engine.runAndWait();
            exit()
        elif inp == "": pass
        else: txt = "Sorry sir, could you phrase your request differently?"; print(txt); engine.say(txt); engine.runAndWait();
            