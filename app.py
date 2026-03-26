from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, abort
import numpy as np, os, openai

from argparse import ArgumentParser

from linebot import (LineBotApi, WebhookHandler)
from linebot.exceptions import (InvalidSignatureError)
from linebot.models import (MessageEvent, TextMessage, TextSendMessage, ImageSendMessage)


from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import  RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings

app = Flask(__name__)

try:
    # 設定 Azure OpenAI 的連線環境變數
    os.environ["OPENAI_API_TYPE"] = "azure"
    os.environ["AZURE_OPENAI_ENDPOINT"] = ""  # 你的連結
    os.environ["AZURE_OPENAI_API_KEY"] = ""  # 你的金鑰
    os.environ["OPENAI_API_VERSION"] = ""  # 你的版本 2024-12-01-preview

    openai.api_type = "azure"
    openai.api_base = os.getenv('AZURE_OPENAI_ENDPOINT')
    openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
    openai.api_version = os.getenv('OPENAI_API_VERSION')

    llm = AzureChatOpenAI(temperature=0, deployment_name="gpt-4o")

    file_path = './static/***'  # 要讀取的 PDF 檔路徑，檔案記得放在 static 資料夾裡
    loader = PyPDFLoader(file_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    texts = loader.load_and_split(splitter)      
    embeddings = AzureOpenAIEmbeddings(model = "text-embedding-3-large", chunk_size = 16)
    vectorstore = FAISS.from_documents(texts, embeddings)
    chat_history = []
    now = (datetime.now() + timedelta(hours=8) ).strftime("%Y/%m/%d %H:%M:%S")  # 取得目前時間，並加上台灣時區（UTC+8）
    with open('/home/logerr0.txt', mode = 'a+') as f:
        f.write('vectorstore done in API on ' + now + "\n")
        f.write("-" * 50 + "\n")
except Exception as e:
        now = (datetime.now() + timedelta(hours=8) ).strftime("%Y/%m/%d %H:%M:%S")
        ai = 'Something wrong in API, try again'
        with open('/home/logerr0.txt', mode = 'a+') as f:
            f.write(str(e) + now + "\n")
            f.write("-" * 50 + "\n")

# LINE Bot 相關參數與物件
channel_access_token = ""  # 長碼
channel_secret = ""  # 短碼
line_bot_api = LineBotApi(channel_access_token)
handler = WebhookHandler(channel_secret)

# 首頁
@app.route('/')
def index():
   print('Request for index page received')
   return render_template('index.html')

# 瀏覽器小圖示
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/hello', methods=['POST'])
def hello():
   name = request.form.get('name')

   if name:
       print('Request for hello page received with name=%s' % name)
       return render_template('hello.html', name = name)
   else:
       print('Request for hello page received with no name or blank name -- redirecting')
       return redirect(url_for('index'))

# 接收 LINE Webhook POST 請求
@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']  # LINE 用來驗證請求是否合法的簽章

    body = request.get_data(as_text=True)
    print("Request body: " + body)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return 'OK'

# LINE Bot 訊息事件處理
@handler.add(MessageEvent, message=TextMessage)
def message_text(event):
    now = (datetime.now() + timedelta(hours=8) ).strftime("%Y/%m/%d %H:%M:%S")
    try:
        qa = ConversationalRetrievalChain.from_llm(llm = llm, retriever = vectorstore.as_retriever())
        ai =  event.message.text
        result = qa({"question": ai, 'chat_history': chat_history })
        aians = result['answer']

    except Exception as e:
        ai = 'Something wrong in API, try again'
        with open('/home/logerr.txt', mode = 'a+') as f:
            f.write(str(e) + "\n")
            f.write(str(event.reply_token)+"\n")
            f.write("-" * 50 + "\n")
    replytext = "RAG : " + aians + ' ...({})'.format(now)
    
    with open('/home/log.txt', mode = 'a+') as f:
        f.write(replytext + "\n")
        f.write(str(event.reply_token)+"\n")
        f.write("-" * 50 + "\n")

    message = [
        TextSendMessage( text = replytext )
    ]

    line_bot_api.reply_message(
        event.reply_token,
        message
    )    

if __name__ == '__main__':    
    app.run()
    