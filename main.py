import os
import flask
from flask import Flask, request, render_template
import pandas as pd
import ktrain
import random
from flask_restful import Resource, Api

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

        # 첫 번째 컬럼을 index로 사용하도록 지정하여 로드(us 데이터만 사용)
        df_us = pd.read_csv('./data/youtube_us.csv', index_col=0)
        # df_kr = pd.read_csv('./data/youtube_kr.csv', index_col = 0)

        # 예측기 load
        predictor = ktrain.load_predictor('./predictor')

        # 감정 확률 예측
        happy, angry, cry = predictor.predict(text, return_proba=True)

        # 감정 예측
        sentiment = predictor.predict(text)

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
    # app.run()