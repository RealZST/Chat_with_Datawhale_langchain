#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   call_llm.py
@Time    :   2023/10/18 10:45:00
@Author  :   Logan Zou 
@Version :   1.0
@Contact :   loganzou0421@163.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   将各个大模型的原生接口封装在一个接口
'''

'''
Unifies API call interface for different LLM platforms.
When 'Chat with LLM' is clicked or the Enter is pressed, the selected LLM is invoked 
through the `get_completion` function in this script to generate responses.
'''

import openai
import json
import requests  # Import requests to call RESTful APIs (for Wenxin)
import _thread as thread
import base64
import datetime
from dotenv import load_dotenv, find_dotenv
import hashlib
import hmac
import os
import queue
from urllib.parse import urlparse
import ssl
from datetime import datetime
from time import mktime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time
import zhipuai
from langchain.utils import get_from_dict_or_env

import websocket  # Import websocket to support WebSocket connections (for Spark)


def get_completion(prompt :str, model :str, temperature=0.1,api_key=None, secret_key=None, access_token=None, appid=None, api_secret=None, max_tokens=2048):
    '''
    Selects the LLM based on the 'model' parameter and generate th response.

    Args:
        prompt: Input prompt
        model: Model name
        secret_key, access_token: Required for Wenxin models
        appid, api_secret: Required for Spark models
        max_tokens: Maximum sequence length to return

    Return:
        Generated response from the model (str)
    '''
    if model in ["gpt-3.5-turbo", "gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-0613", "gpt-4", "gpt-4-32k"]:
        return get_completion_gpt(prompt, model, temperature, api_key, max_tokens)
    elif model in ["ERNIE-Bot", "ERNIE-Bot-4", "ERNIE-Bot-turbo"]:
        return get_completion_wenxin(prompt, model, temperature, api_key, secret_key)
    elif model in ["Spark-1.5", "Spark-2.0"]:
        return get_completion_spark(prompt, model, temperature, api_key, appid, api_secret, max_tokens)
    elif model in ["chatglm_pro", "chatglm_std", "chatglm_lite"]:
        return get_completion_glm(prompt, model, temperature, api_key, max_tokens)
    else:
        return "Invalid model"
    
def get_completion_gpt(prompt : str, model : str, temperature : float, api_key:str, max_tokens:int):
    # Creates a wrapper for OpenAI's GPT API
    if api_key == None:
        api_key = parse_llm_api_key("openai")
    openai.api_key = api_key
    messages = [{"role": "user", "content": prompt}]
    # Call OpenAI API
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens = max_tokens,
    )

    return response.choices[0].message["content"]

def get_access_token(api_key, secret_key):
    """
    Retrieves the access_token using API Key and Secret Key.
    """
    url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={api_key}&client_secret={secret_key}"
    payload = json.dumps("")
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    
    return response.json().get("access_token")

def get_completion_wenxin(prompt : str, model : str, temperature : float, api_key:str, secret_key : str):
    # Creates a wrapper for Wenxin API
    if api_key == None or secret_key == None:
        api_key, secret_key = parse_llm_api_key("wenxin")
    access_token = get_access_token(api_key, secret_key)
    url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/eb-instant?access_token={access_token}"
    payload = json.dumps({
        "messages": [
            {
                "role": "user",# user prompt
                "content": "{}".format(prompt)# 输入的 prompt
            }
        ]
    })
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    js = json.loads(response.text)

    return js["result"]

def get_completion_spark(prompt : str, model : str, temperature : float, api_key:str, appid : str, api_secret : str, max_tokens : int):
    # Creates a wrapper for Spark API
    if api_key == None or appid == None and api_secret == None:
        api_key, appid, api_secret = parse_llm_api_key("spark")
    
    if model == "Spark-1.5":
        domain = "general"  
        Spark_url = "ws://spark-api.xf-yun.com/v1.1/chat" 
    else:
        domain = "generalv2" 
        Spark_url = "ws://spark-api.xf-yun.com/v2.1/chat"

    question = [{"role":"user", "content":prompt}]
    response = spark_main(appid,api_key,api_secret,Spark_url,domain,question,temperature, max_tokens)
    return response

def get_completion_glm(prompt : str, model : str, temperature : float, api_key:str, max_tokens : int):
    # Creates a wrapper for Zhipuai API
    if api_key == None:
        api_key = parse_llm_api_key("zhipuai")
    zhipuai.api_key = api_key

    response = zhipuai.model_api.invoke(
        model=model,
        prompt=[{"role":"user", "content":prompt}],
        temperature = temperature,
        max_tokens=max_tokens
        )

    return response["data"]["choices"][0]["content"].strip('"').strip(" ")


# WebSocket-based API for Spark
answer = ""

class Ws_Param(object):
    # 初始化
    def __init__(self, APPID, APIKey, APISecret, Spark_url):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.host = urlparse(Spark_url).netloc
        self.path = urlparse(Spark_url).path
        self.Spark_url = Spark_url
        # 自定义
        self.temperature = 0
        self.max_tokens = 2048

    # 生成url
    def create_url(self):
        # 生成RFC1123格式的时间戳
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        # 拼接字符串
        signature_origin = "host: " + self.host + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + self.path + " HTTP/1.1"

        # 进行hmac-sha256进行加密
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()

        signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding='utf-8')

        authorization_origin = f'api_key="{self.APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'

        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')

        # 将请求的鉴权参数组合为字典
        v = {
            "authorization": authorization,
            "date": date,
            "host": self.host
        }
        # 拼接鉴权参数，生成url
        url = self.Spark_url + '?' + urlencode(v)
        # 此处打印出建立连接时候的url,参考本demo的时候可取消上方打印的注释，比对相同参数时生成的url与自己代码生成的url是否一致
        return url

# 收到websocket错误的处理
def on_error(ws, error):
    print("### error:", error)

# 收到websocket关闭的处理
def on_close(ws,one,two):
    print(" ")

# 收到websocket连接建立的处理
def on_open(ws):
    thread.start_new_thread(run, (ws,))

def run(ws, *args):
    data = json.dumps(gen_params(appid=ws.appid, domain= ws.domain,question=ws.question, temperature = ws.temperature, max_tokens = ws.max_tokens))
    ws.send(data)

# 收到websocket消息的处理
def on_message(ws, message):
    # print(message)
    data = json.loads(message)
    code = data['header']['code']
    if code != 0:
        print(f'请求错误: {code}, {data}')
        ws.close()
    else:
        choices = data["payload"]["choices"]
        status = choices["status"]
        content = choices["text"][0]["content"]
        print(content,end ="")
        global answer
        answer += content
        # print(1)
        if status == 2:
            ws.close()

def gen_params(appid, domain,question, temperature, max_tokens):
    data = {
        "header": {
            "app_id": appid,
            "uid": "1234"
        },
        "parameter": {
            "chat": {
                "domain": domain,
                "random_threshold": 0.5,
                "max_tokens": max_tokens,
                "temperature" : temperature,
                "auditing": "default"
            }
        },
        "payload": {
            "message": {
                "text": question
            }
        }
    }
    return data

def spark_main(appid, api_key, api_secret, Spark_url,domain, question, temperature, max_tokens):
    output_queue = queue.Queue()
    def on_message(ws, message):
        data = json.loads(message)
        code = data['header']['code']
        if code != 0:
            print(f'请求错误: {code}, {data}')
            ws.close()
        else:
            choices = data["payload"]["choices"]
            status = choices["status"]
            content = choices["text"][0]["content"]
            output_queue.put(content)
            if status == 2:
                ws.close()

    wsParam = Ws_Param(appid, api_key, api_secret, Spark_url)
    websocket.enableTrace(False)
    wsUrl = wsParam.create_url()
    ws = websocket.WebSocketApp(wsUrl, on_message=on_message, on_error=on_error, on_close=on_close, on_open=on_open)
    ws.appid = appid
    ws.question = question
    ws.domain = domain
    ws.temperature = temperature
    ws.max_tokens = max_tokens
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
    return ''.join([output_queue.get() for _ in range(output_queue.qsize())])


def parse_llm_api_key(model:str, env_file:dict()=None):
    """
    Parses API keys based on the model name 
    """   
    if env_file == None:
        _ = load_dotenv(find_dotenv())
        env_file = os.environ
    
    if model == "openai":
        return env_file["OPENAI_API_KEY"]
    elif model == "wenxin":
        return env_file["wenxin_api_key"], env_file["wenxin_secret_key"]
    elif model == "spark":
        return env_file["spark_api_key"], env_file["spark_appid"], env_file["spark_api_secret"]
    elif model == "zhipuai":
        return get_from_dict_or_env(env_file, "zhipuai_api_key", "ZHIPUAI_API_KEY")
    else:
        raise ValueError(f"model{model} not support!!!")
