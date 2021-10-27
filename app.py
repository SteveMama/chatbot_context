from flask import Flask, request, render_template
from flask_ngrok import run_with_ngrok
import speech_recognition as sr
import aiml
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import tflearn
import nltk
from nltk.stem.lancaster import LancasterStemmer





stemmer = LancasterStemmer()


#deploying the hate speech model
tokenizer = AutoTokenizer.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain")
model = AutoModelForSequenceClassification.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain")
classifier = pipeline('text-classification', model=model, tokenizer = tokenizer)


global rec_text
rec_text=''





#Flasking the application
app = Flask(__name__)
run_with_ngrok(app)

#the brain
model.load("C://Users//prash//Desktop//Anorak//SOE_GUI//models//model.tflearn")
with open("C://Users//prash//Desktop//Anorak//SOE_GUI//intents.json") as json_data:
    intents = json.load(json_data)



def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words= [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=False):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("Found in bag: %s"% w)
    return(np.array(bag))




context = {}
ERROR_THRESHOLD = 0.25
def classify(sentence):
    results = model.predict([bow(sentence, words)])[0]
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    return return_list

def response(sentence, userID='123', show_details=False):
    results = classify(sentence)
    if results:
        while results:
            for i in intents['intents']:
                if i['tag'] == results[0][0]:
                    if 'context_set' in i:
                        if show_details: print ('context:', i['context_set'])
                        context[userID] = i['context_set']
                    if not 'context_filter' in i or \
                            (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                        if show_details: print ('tag:', i['tag'])
                        return print(random.choice(i['responses']))
            results.pop(0)


#deepfake function
def display_video(omh):
    text = omh
    #what is statue of equality" or "tell me about statue of equality" or "tell me about the statue of ramanuja
    if text == "SOE-1":
        path = "A1.mp4"
        #who is ramanujacharya" or "who is ramanuja acharya
    elif text == "SOE-2":
        path = "A2.mp4"
        #what are the social reforms ramanuja introduced
    elif text == "SOE-3":
        path = "A3.mp4"
        #when will the statue of equality be open for public
    elif text == "SOE-4":
        path = "A4.mp4"
    else :
        path = "A1.mp4"
    return path



r = sr.Recognizer()

#INPUT AUDIO -> HATE SPEECH
def micro(audio_text):
    try:
        print("Text: "+r.recognize_google(audio_text))
        rec_text = r.recognize_google(audio_text, language='en-US').lower()
        hello  = classifier(rec_text)
        for h in hello:
            g = h['score']
            print(h['score'])
        if g>=0.58:
            doge = display_video(rec_text)
        else:
            doge = "SOE-3"

    except:
        doge = "Sorry, I did not get that"
        print("Sorry, I did not get that")
    return doge


@app.route('/', methods=['POST'])
def my_form_post():
    with sr.Microphone() as source:
        print("Ask the Acharya")
        audio_text = r.listen(source)
        print("Question recorded, thanks!")
    omh = micro(audio_text)

    rec_text = omh
    return render_template('form.html',text1=rec_text, filename=filename)


if __name__ == "__main__":
    app.run()

