import string
import pandas as pd
import matplotlib.pyplot as plt

# 日本語のレビューのみを抽出する関数
def filter_by_ascii_rate(text,threshold=0.9):
    ascill_letters = set(string.printable)
    rate = sum(c in ascill_letters for c in text)/len(text)
    return rate <= threshold

# データセットを読み込む関数
def load_dataset(filename,n=5000,state=6):
    df = pd.read_csv(filename,sep='\t')
    
    mapping = {1:0,2:0,4:1,5:1}
    df = df[df.star_rating != 3]
    df.star_rating = df.star_rating.map(mapping)
    
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

# # 学習と評価を行う関数
# def train_and_eval(X_train,y_train,X_test,y_test,
#                    lowercase=False,tokenize=None,preprocessor=None):
    
#     #　与えられたテキストデータをベクトル化
#     vectorizer = CountVectorizer(lowercase=lowercase,
#                                  tokenizer=tokenize,
#                                  preprocessor=preprocessor)
#     X_train_vec = vectorizer.fit_transform(X_train)
#     X_test_vec = vectorizer.transform(X_test)
    
#     #　学習
#     # ロジスティック回帰を使用して、テキストデータを分類する
#     clf = LogisticRegression(solver='liblinear')
#     clf.fit(X_train_vec,y_train)
    
#     #　評価
#     # テストデータのクラスラベルを予測し、予測の正確度を計算する
#     y_pred = clf.predict(X_test_vec)
#     score = accuracy_score(y_test,y_pred)
#     print('{:.4f}'.format(score))


def train_and_eval(X_train,y_train,X_test,y_test,vectorizer):
    x_train_vec = vectorizer.fit_transform(X_train)
    x_test_vec = vectorizer.transform(X_test)
    clf = LogisticRegression(solver='liblinear')
    clf.fit(x_train_vec,y_train)
    y_pred = clf.predict(x_test_vec)
    score = accuracy_score(y_test,y_pred)
    print('{:.4f}'.format(score))
    
def plot_history(history):
    #設定
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    epochs = range(1,len(loss)+1)
    
    #Plotting loss
    plt.plot(epochs,loss,'r',label='Training loss')
    plt.plot(epochs,val_loss,'b',label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.figure()
    
    #Plotting accuracy
    plt.plot(epochs,acc,'r',label='Training acc')
    plt.plot(epochs,val_acc,'b',label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    
    plt.show()