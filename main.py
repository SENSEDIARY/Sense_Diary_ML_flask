import os
import flask
from flask import Flask, request, render_template
from flask_restful import Resource, Api

import pandas as pd
import ktrain
import random

import re

app = Flask(__name__)
api = Api(app)

# 메인 페이지 라우팅
@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')

#데이터 예측 처리
@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method == 'POST':

        #업로드 파일 처리 분기
        # text = request.data
        text = request.form['text']
        if not text: return render_template('index.html', result = 'No result')

        # # 텍스트 전처리
        #
        # # Defining dictionary containing all emojis with their meanings.
        emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad',
                  ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
                  ':-@': 'shocked', ':@': 'shocked', ':-$': 'confused', ':\\': 'annoyed',
                  ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
                  '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
                  '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink',
                  ';-)': 'wink', 'O:-)': 'angel', 'O*-)': 'angel', '(:-D': 'gossip', '=^.^=': 'cat'}

        ## Defining set containing all stopwords in english.
        stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
                        'and', 'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before',
                        'being', 'below', 'between', 'both', 'by', 'can', 'd', 'did', 'do',
                        'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from',
                        'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
                        'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
                        'into', 'is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
                        'me', 'more', 'most', 'my', 'myself', 'now', 'o', 'of', 'on', 'once',
                        'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'own', 're',
                        's', 'same', 'she', "shes", 'should', "shouldve", 'so', 'some', 'such',
                        't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
                        'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
                        'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was',
                        'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom',
                        'why', 'will', 'with', 'won', 'y', 'you', "youd", "youll", "youre",
                        "youve", 'your', 'yours', 'yourself', 'yourselves']

        # # stopword
        # stopword_nltk = stopwords.words('english')
        #
        # # Create Lemmatizer and Stemmer.
        # wordLemm = WordNetLemmatizer()
        #
        # Defining regex patterns.
        urlPattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
        userPattern = '@[^\s]+'
        alphaPattern = "[^a-zA-Z0-9]"
        sequencePattern = r"(.)\1\1+"
        seqReplacePattern = r"\1\1"

        # 소문자 변환
        text = text.lower()

        # URL 전처리
        text = re.sub(urlPattern, ' URL', text)

        # emojis 전처리
        for emoji in emojis.keys():
            text = text.replace(emoji, "EMOJI" + emojis[emoji])

        # Usernames 전처리
        text = re.sub(userPattern, ' USER', text)

        # 알파벳 아닌 문자 제거
        text = re.sub(alphaPattern, " ", text)

        # 3번 이상 반복되는 문자 2개짜리 문자로 변환
        text = re.sub(sequencePattern, seqReplacePattern, text)

        preprocess_text = ''
        for word in text.split():
            # Checking if the word is a stopword.
            if word not in stopwordlist:
                if len(word) > 1:
                    # Lemmatizing the word.
                    # word = wordLemm.lemmatize(word)

                    preprocess_text += (word + ' ')

        # 첫 번째 컬럼을 index로 사용하도록 지정하여 로드(us 데이터만 사용)
        df_us = pd.read_csv('./data/youtube_us.csv', index_col=0)
        # df_kr = pd.read_csv('./data/youtube_kr.csv', index_col = 0)

        # 예측기 load
        predictor = ktrain.load_predictor('./predictor')

        # 감정 확률 예측
        happy, angry, cry = predictor.predict(preprocess_text, return_proba=True)

        # 감정 예측
        sentiment = predictor.predict(preprocess_text)

        # 감정 별 유튜브 데이터프레임 분리
        df_us_happy = df_us[df_us['sentiment'] == 'smile']
        # df_us_angry = df_us[df_us['sentiment'] == 'angry']
        df_us_cry = df_us[df_us['sentiment'] == 'sob']

        # 난수 활용 임의 영상
        if sentiment in ['smile', 'angry']:  # 기쁨, 화남
            num = random.randrange(0, df_us_happy.shape[0])
            vid = df_us_happy.loc[num, 'video_id']

        else:  # 슬픔
            num = random.randrange(0, df_us_cry.shape[0])
            vid = df_us_cry.loc[num, 'video_id']

        # 결과 list [기쁨 확률, 화남 확률, 슬픔 확률, 감정 라벨, 유튜브 id]
        result = [happy, angry, cry, sentiment, vid]

        return render_template('index.html', result = result)

if __name__ == '__main__':
    # Flask 서비스 스타트
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
    # app.run(host='0.0.0.0', port=8000, debug=True)