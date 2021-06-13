from flask import Flask, render_template, request
import speech_recognition as sr
import os
from werkzeug.utils import secure_filename
import pandas as pd
import nltk
import math
from io import StringIO as stio
from nltk.sentiment.vader import SentimentIntensityAnalyzer
app = Flask(__name__)
@app.route('/')
def upload_file():
    return render_template('upload.html')
@app.route('/uploader', methods=['GET', 'POST'])
def uploader():
    if request.method == 'POST':
        f = request.files['f']
        text_input = request.form['text_input']
        mic_input = request.form['mic_input']
        if text_input == "" and mic_input =="" and f.filename != "":
            #print(s)
            f.save(secure_filename(f.filename))
            #print(f.filename)
            fn = secure_filename(f.filename)
            new_path = os.path.abspath(fn)
            p = new_path
            AUDIO_FILE = (p)
            r = sr.Recognizer()
            with sr.AudioFile(AUDIO_FILE) as source:
                audio = r.record(source)
            try:
                # print("audio file contains"+ r.recognize_google(audio))
                st = ""
                st = st + r.recognize_google(audio)
                print(st)
            except sr.UnknownValueError:
                print("GSR could not understand audio")
            except sr.RequestError:
                print("Google Couldn't get results")
            print(new_path)
        elif mic_input!="" and text_input =="" and f.filename == "":
            st = mic_input
        elif text_input !="" and mic_input == "" and f.filename == "":
            st = text_input
        else:
            return render_template('upload.html')

        StringData = stio(st)
        dfs = pd.read_csv(StringData , lineterminator="|", engine="c")
        #dfs = pd.read_csv(StringData, sep=".")
        # dfs=list(dfs['Timeline'])
        sid = SentimentIntensityAnalyzer()
        negv, neuv, posv, c = 0, 0, 0, 0
        for data in dfs:
            if (data == ""):
                break
            c += 1
            ss = sid.polarity_scores(data)
            for k in ss:
                if (k == 'pos'):
                    posv += ss[k]
                if (k == 'neu'):
                    neuv += ss[k]
                if (k == 'neg'):
                    negv += ss[k]
        ne = negv / c
        ne = math.ceil(ne*100)
        neut = neuv / c
        neut = math.ceil(neut*100)
        p = 100 - ne - neut
        return render_template('about.html', st=st, negv=ne, neuv=neut, posv=p)
if __name__ == '__main__':
    app.run(debug=True)
