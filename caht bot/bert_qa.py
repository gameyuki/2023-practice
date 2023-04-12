from transformers import BertForQuestionAnswering,AutoTokenizer
import tkinter as tk
import torch  

#　文字の入力
input_text = input('質問を入力してください：')

# モデルの選択
config='bert_config.json' # modelの設定
t_config='tokenizer_config.json' # tokenizerの設定
model = BertForQuestionAnswering.from_pretrained('pytorch_model.bin',config=config,ignore_mismatched_sizes=True) # 学習済みモデルの選択
torch.save(model.state_dict(),'model.pth') # モデルを一度保存してキーに関するエラーが起こらないようにする
model.load_state_dict(torch.load('model.pth',map_location=torch.device('cpu'))) # 学習済みモデルの読み込み
tokenizer = AutoTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking',config=t_config) # tokenizerの読み込み

def predict(question,context):
    input_ids=tokenizer.encode(question,context) # tokenizerで形態素解析しつつコードに変換する
    output= model(torch.tensor([input_ids])) # 学習済みモデルを用いて解析
    all_tokens = tokenizer.convert_ids_to_tokens(input_ids) # コード化した文章を復号化する
    prediction = ''.join(all_tokens[torch.argmax(output.start_logits) : torch.argmax(output.end_logits)+1]) # 答えに該当する部分を抜き取る
    prediction = prediction.replace("#", "") # 余分な文字を削除する
    prediction = prediction.replace(" ","")
    prediction = prediction.replace("[SEP]","") 
    return prediction

def bert():
#入力する文章
    # 質問を表示
    print(input_text) 
    # contextとなる文章をファイルから読み込む
    with open('bert_qa.txt',encoding='utf-8') as f:
        context=f.read()
        context=context.replace('\n','') # 改行を削除
    prediction=predict(input_text,context) # 答えを取得
    print(prediction)


# メイン関数
def main():
    while(1):
        input_text = input('質問を入力してください：')
        bert()
    

if __name__ == "__main__":
    main()