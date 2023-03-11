import string
import pandas as pd

# 日本語のレビューのみを抽出する関数
def filter_by_ascii_rate(text,threshold=0.9):
    ascill_letters = set(string.printable)
    rate = sum(c in ascill_letters for c in text)/len(text)
    return rate <= threshold

# データセットを読み込む関数
def load_dataset(filename,n=5000,state=6):
    df = pd.read_csv(filename,sep='\t')
    
    # extracts japanese text.
    is_jp = df.review_body.apply(filter_by_ascii_rate)
    df = df[is_jp]
    
    #sampling
    #　サンプリングとは、データセットからランダムに一部を抽出すること
    df = df.sample(frac=1,random_state=state)
    grouped = df.groupby('star_rating')
    df = grouped.head(n=n)
    return df.review_body.values,df.star_rating.values

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 学習と評価を行う関数
def train_and_eval(X_train,y_train,X_test,y_test,
                   lowercase=False,tokenize=None,preprocessor=None):
    
    #　与えられたテキストデータをベクトル化
    vectorizer = CountVectorizer(lowercase=lowercase,
                                 tokenizer=tokenize,
                                 preprocessor=preprocessor)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    #　学習
    # ロジスティック回帰を使用して、テキストデータを分類する
    clf = LogisticRegression(solver='liblinear')
    clf.fit(X_train_vec,y_train)
    
    #　評価
    # テストデータのクラスラベルを予測し、予測の正確度を計算する
    y_pred = clf.predict(X_test_vec)
    score = accuracy_score(y_test,y_pred)
    print('{:.4f}'.format(score))