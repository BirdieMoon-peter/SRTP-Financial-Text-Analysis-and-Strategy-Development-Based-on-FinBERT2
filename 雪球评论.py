"""
Author: Peter Li
Date: 2025-11-09 16:17:37
Description: 文件描述
"""

import pandas as pd
from bs4 import BeautifulSoup
import re
import requests
import json
import warnings
warnings.filterwarnings("ignore")

def download_page(url,para = None):
    headers = {
            
            
      'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)'
               ' Chrome/91.0.4472.114 Safari/537.36 Edg/91.0.864.59'}
    if para:
        response = requests.get(url,params=para,headers = headers)
    else:
        response = requests.get(url,headers = headers)  
    response.encoding = response.apparent_encoding
    if response.status_code == 200:
        return response.text
    else:
        print ("failed to download the page") 
        
        
def xueqiu(Start,End):   #两个参数分别表示开始读取与结束读取的页码
    comments_list = []  
    #将响应头的参数补齐以避免无法读取的情况
    headers = {
            
            
      "Refer":"https://xueqiu.com/k?q=%E9%99%90%E5%94%AE%E8%82%A1%E8%A7%A3%E7%A6%81",
           "Host": "xueqiu.com",
           "Cookie":"acw_tc=2760820216245210669791692ead10af083a4c98f7b2541837fa80a96e6849; xq_a_token=f257b9741beeb7f05f6296e58041e56c810c8ef8; xqat=f257b9741beeb7f05f6296e58041e56c810c8ef8; xq_r_token=2e05f6c50e316248a8a08ab6a47bc781da7fddfb; xq_id_token=eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJ1aWQiOi0xLCJpc3MiOiJ1YyIsImV4cCI6MTYyNjQwMzgwNSwiY3RtIjoxNjI0NTIxMDUxMDg4LCJjaWQiOiJkOWQwbjRBWnVwIn0.YuyB7t3x8jCpO5aenapczHmoXzYznlC9XUntALPpBV8pBEZIBi1LU8oltfyvxRMerCY3VqsBR8moa64fSvxzArV0RMuL7633bjcB-b0GrQY3tvsva0Nlfj7w3tRTavMfw04fU_LruFbHhoc-LR-D83lH7e_Ndp4ZmwIayI3SEARBHqDWa4RjZ-KAxLiQ-hnS8usiodS8cxyTrmNtcr0hLB59zPCRq2KzO3RCVFuYmaNIRyWXXqcmjFS3tvpQ4FlOLC4YVOzqlb-vyhWJAAuQTXj7-z6XnQcRHRNQw53WRmiivgzv3YVPqIq0qQslJjIczAmmTeZxqYEy3ZMan3Bwow; u=441624521066981; Hm_lvt_1db88642e346389874251b5a1eded6e3=1624521071; Hm_lpvt_1db88642e346389874251b5a1eded6e3=1624521079; device_id=24700f9f1986800ab4fcc880530dd0ed",
           'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.140 Safari/537.36 Edge/17.17134'}
     #遍历每一个URL
    for i in range(int(Start),int(End)+1):
        url='https://xueqiu.com/query/v1/search/status.json?sortId=3&q=%E9%99%90%E5%94%AE%E8%82%A1%E8%A7%A3%E7%A6%81&count=10''&page='+ str(i)
       # print(url)    用于检查
        response = requests.get(url, headers=headers, verify=False, timeout=30)   # 禁止重定向
        content = response.text
        #读取的是json文件。因此就用json打开啦
        result = json.loads(content)  
        #找到原始页面中数据所在地
        comments = result['list']                
        for i in range (0,len(comments)):            
            comment={}            
            #取出需要的字段并存入字典中
            comment['time']=comments[i]['timeBefore']            
            comment['target']=comments[i]['target']  
            #从初始页面获取内容的链接（不全，需要自行补齐），直接调用自己的方法读取文本内容
            comment['text'] = get_text("https://xueqiu.com"+comments[i]['target'])
            comments_list.append(comment)               
    return comments_list

def get_text(url):       
    soup=BeautifulSoup(download_page(url))
    pattern = re.compile("article__bd__detail.*?")#按标签寻找
    all_comments = soup.find_all("div",{
            
            
      'class':pattern})
    text1=all_comments[0]
    con=text1.get_text()#只提取文字   
    return con

def output_csv(datalist):
    print(type(datalist),len(datalist))  # <class 'list'> 100用于检查
    import csv
    csv_file = open("comments_data.csv", 'a+', newline='', encoding='utf-8-sig')  # 解决中文乱码问题。a+表示向csv文件追加
    writer = csv.writer(csv_file)
    writer.writerow(['Date', 'URL', 'Content'])
    for data in datalist:
        writer.writerow([data['time'], "https://xueqiu.com"+data['target'],data['text'],])#原来的链接不全因此给他补齐
    csv_file.close()

if __name__=="__main__":
    Start = input('请输入开始爬取的页数：') 
    End = input('请输入结束爬取的页数：')
    result = xueqiu(Start,End)
    output_csv(result)
