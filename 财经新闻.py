import threading #多线程操作
import urllib
import time
import requests,re,json,time,os
import lxml #一个Python库,使用它可以轻松处理XML和HTML文件,还可以用于web爬取
from lxml import etree
from matplotlib import pyplot
import matplotlib
import matplotlib.pyplot as plt #数据可视化

pyplot.rcParams['font.sans-serif'] = ['SimHei']  # 图形参数设置，用来正常显示中文标签，国标黑体
pyplot.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

#得到最大页码的数值
def getallpage(url):
    pagedata = requests.get(url).content.decode("gbk")
    # print(pagedata)
    # lxml教程：https://www.cnblogs.com/zhangxinqi/p/9210211.html#_label2
    mytree = lxml.etree.HTML(pagedata)
    #取所有的页码数
    if pagedata.find("page_newslib"):
        data = mytree.xpath("//*[@class=\"page_newslib\"]//a[last()-1]/text()")
        return data
    else:
        return ['1']

#得到页面的内容并保存
def everypagecontent(url,number):
    #解决服务器延时的问题try
    try:
        pagedata = requests.get(url).content.decode("gbk")
        mytree = lxml.etree.HTML(pagedata)
        # print(pagedata)
        # 取所有的内容
        datas = mytree.xpath("//*[@class = \"newlist\"]//li/span/a/text()")
        for data in datas:
            data= data+"\r\n"
            with open(number+".txt","a") as file:
                file.write(data)
                file.flush()
        return datas
    except:
        print("服务器超时")


def getpageurl(url):
    pagenumber = getallpage(url)[0]
    # print(pagenumber)
    for i in range(1, int(pagenumber) + 1):
        try:
            if i == 1:
                url = "http://stock.jrj.com.cn/share," + number + ",ggxw.shtml"
            else:
                url = "http://stock.jrj.com.cn/share," + number + ",ggxw_" + str(i) + ".shtml"
        except:
            pass
        print(everypagecontent(url, number))

numbers = ['601011']
# numbers = ['601011',……](可以将自己喜欢的多个股票填入……区域)

#得到股票名称，获取新闻信息生成文件
for number in numbers:
    print(number)
    url = "http://stock.jrj.com.cn/share,"+number+",ggxw.shtml"
    getpageurl(url)

