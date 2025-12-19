#首先引入第三方库
import requests
import re
import openpyxl
import time

#定义一个访问接口的函数
def get_report(page_num, date, trade="金融业;科学研究和技术服务业;制造业"):
    url = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
    headers = {
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
        "Content-Length": "195",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "Host": "www.cninfo.com.cn",
        "Origin": "http://www.cninfo.com.cn",
        "Proxy-Connection": "keep-alive",
        "Referer": "http://www.cninfo.com.cn/new/commonUrl/pageOfSearch?url=disclosure/list/search&checkedCategory=category_ndbg_szsh",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36 Edg/113.0.1774.42",
        "X-Requested-With": "XMLHttpRequest"
    }
    '''
    参数信息
     plate: sz;sh, 表示沪深两市
     seDate：查询时间
     trade: 企业类别
    '''
    data = {
        "pageNum": page_num,
        "pageSize": 30,
        "column": "szse",
        "tabName": "fulltext",
        "plate": "sz;sh",
        "searchkey": "",
        "secid": "",
        "category": "category_ndbg_szsh",
        "trade": trade,
        "seDate": date,
        "sortName": "code",
        "sortType": "asc",
        "isHLtitle": "false"
    }
    response = requests.post(url, data=data, headers=headers)
    return response


def downlaod_report(date, trade="金融业;科学研究和技术服务业;制造业"):
    global counter
    all_results = []
    page_num = 1
    response_test = get_report(page_num, date, trade)
    data_test = response_test.json()
    total_pages = data_test["totalpages"]
    max_retries = 3 #最大重试次数
    retry_count = 0 #当前重试次数
    while page_num <= total_pages:
        response = None

        # 重试机制
        while retry_count <= max_retries:
            # 发送请求
            try:
                # response = requests.post(url, data=data,headers=headers)
                response = get_report(page_num, date, trade)
                response.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                print(f"出现错误！: {e}")
                print(f"5秒后重试...")
                time.sleep(5)
                retry_count += 1

        if retry_count > max_retries:
            print(f"{max_retries} 次重试后均失败. 跳过第 {page_num}页.")
            page_num += 1
            retry_count = 0
            continue
        else:
            # 解析数据
            try:
                data = response.json()
                # print(f"正在下载第 {page_num}/{total_pages} 页")
                print(f"\r正在下载第 {counter}/{sum} 页",end='')
                # 尝试解析公告数据，如果解析失败则重试
                retry_count = 0
                while True:
                    try:
                        if data["announcements"] is None:
                            raise Exception("公告数据为空")
                        else:
                            all_results.extend(data["announcements"])
                        break
                    except (TypeError, KeyError) as e:
                        print(f"解析公告数据失败: {e}")
                        print(f"5秒后重试...")
                        time.sleep(5)
                        retry_count += 1
                        if retry_count > max_retries:
                            raise Exception("达到最大重试次数，跳过此页")
                        continue
                page_num += 1
                counter +=1
            except (ValueError, KeyError) as e:
                print(f"解析响应数据失败: {e}")
                print(f"5秒后重试...")
                time.sleep(5)
                retry_count += 1
                if retry_count > max_retries:
                    raise Exception("达到最大重试次数，跳过此页")
                continue
    return all_results


def main(year):
    # 计数器
    global sum
    date_count = f"{year}-01-01~{year}-04-30"
    response = get_report(1, date_count)
    data = response.json()
    sum = data["totalpages"]
    year = year+1
    all_results = []
    time_segments = [
        f"{year}-01-01~{year}-04-01",
        f"{year}-04-02~{year}-04-15",
        f"{year}-04-16~{year}-04-22",
        f"{year}-04-23~{year}-04-26",
        f"{year}-04-27~{year}-04-28",
        f"{year}-04-29~{year}-04-30"
    ]
    
    # 定义企业类别列表
    trade_categories = ["金融业", "科学研究和技术服务业", "制造业"]
    
    for i in time_segments:
        for trade in trade_categories:
            print(f"\n正在获取 {trade} 类别数据...")
            results = downlaod_report(i, trade)
            # 为每个结果添加企业类别信息
            for item in results:
                item["trade_category"] = trade
            all_results.extend(results)

    # 创建Excel文件并添加表头
    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    worksheet.title = "年报链接"
    # 添加企业类别列
    worksheet.append(["公司代码", "公司简称", "标题", "年份", "企业类别", "年报链接"])

    # 解析搜索结果并添加到Excel表格中
    for item in all_results:
        company_code = item["secCode"]
        company_name = item["secName"]
        title = item["announcementTitle"].strip()
        # 剔除不需要的样式和特殊符号，并重新组合标题
        title = re.sub(r"<.*?>", "", title)
        title = title.replace("：", "")
        title = f"《{title}》"

        adjunct_url = item["adjunctUrl"]
        year = re.search(r"\d{4}", title)
        if year:
            year = year.group()
        else:
            year = setYear
        time = f"{year}"
        announcement_url = f"http://static.cninfo.com.cn/{adjunct_url}"
        
        # 获取企业类别
        trade_category = item.get("trade_category", "")

        # 检查标题是否包含排除关键词
        exclude_flag = False
        for keyword in exclude_keywords:
            if keyword in title:
                exclude_flag = True
                break

        # 如果标题不包含排除关键词，则将搜索结果添加到Excel表格中
        if not exclude_flag:
            worksheet.append([company_code, company_name, title, time, trade_category, announcement_url])
    workbook.save(f"年报链接_{setYear}.xlsx")
    
    
if __name__ == '__main__':
    # 全局变量
    # 排除列表可以加入'更正后','修订版'来规避数据重复或公司发布之前年份的年报修订版等问题，
    exclude_keywords = ['英文', '摘要','已取消','公告']
    global counter
    global sum
    counter = 1  # 计数器
    setYear = 2025 #设置下载年份
    Flag = True #是否开启批量下载模式
    if Flag:
        for setYear in range(2018,2026):
            counter = 1  # 计数器
            main(setYear)
            print(f"----{setYear}年下载完成")
    else:
        main(setYear)
        print(f"----{setYear}年下载完成")