"""
Author: Peter Li
Date: 2025-11-09 16:17:55
Description: 年报下载
"""

import pandas as pd
import requests
import os
import glob
import time
import re
from urllib.parse import urlparse

def download_pdfs_from_excel(excel_file, output_base_dir="PDF文件", max_downloads=None):
    """
    从Excel文件中下载PDF文件
    :param excel_file: Excel文件路径
    :param output_base_dir: PDF文件保存的基础目录
    :param max_downloads: 最大下载数量，None表示下载全部
    :return: 下载成功的文件列表
    """
    # 读取Excel文件
    try:
        df = pd.read_excel(excel_file, dtype={'公司代码': str})
        print(f"读取Excel文件: {excel_file}, 包含 {len(df)} 条记录")
    except Exception as e:
        print(f"读取Excel文件失败: {str(e)}")
        return []
    
    # 检查是否有年报链接列
    if '年报链接' not in df.columns:
        print("Excel文件中没有'年报链接'列")
        return []
    
    # 从文件名中提取年份
    file_name = os.path.basename(excel_file)
    year_match = re.search(r'(\d{4})', file_name)
    year = year_match.group(1) if year_match else "未知年份"
    
    # 检查是否有企业类别列
    has_trade_category = '企业类别' in df.columns
    
    # 设置请求头
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36"
    }
    
    # 下载计数器
    download_count = 0
    success_count = 0
    failed_count = 0
    success_files = []
    
    # 遍历每一行数据
    for index, row in df.iterrows():
        # 检查是否达到最大下载数
        if max_downloads is not None and download_count >= max_downloads:
            break
            
        download_count += 1
        
        # 获取公司信息和PDF链接
        company_code = row.get('公司代码', f'未知代码{index}')
        company_name = row.get('公司简称', f'未知公司{index}')
        title = row.get('标题', f'未知标题{index}')
        pdf_url = row.get('年报链接', '')
        
        # 获取企业类别
        if has_trade_category:
            trade_category = row.get('企业类别', '未知类别')
        else:
            trade_category = '未分类'
        
        if not pdf_url:
            print(f"第 {index+1} 行: 没有PDF链接")
            failed_count += 1
            continue
        
        try:
            # 创建目录结构：企业类别/年份
            # 清理企业类别名称，移除不合法的文件名字符
            clean_trade_category = re.sub(r'[\\/*?:"<>|]', '', trade_category)
            output_dir = os.path.join(output_base_dir, clean_trade_category, f"{year}年年报")
            
            # 创建目录
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"创建目录: {output_dir}")
            
            # 从URL中提取文件名
            parsed_url = urlparse(pdf_url)
            clean_title = re.sub(r'[\\/*?:"<>|]', '', f"{company_code}_{company_name}_{year}.pdf")
            file_name = clean_title
            
            file_path = os.path.join(output_dir, file_name)
            
            # 检查文件是否已存在
            if os.path.exists(file_path):
                print(f"第 {index+1} 行: 文件已存在，跳过下载 - {file_name}")
                success_count += 1
                success_files.append(file_path)
                continue
            
            print(f"第 {index+1} 行: 正在下载 - {clean_trade_category}/{year}年年报/{file_name}")
            
            # 发送请求下载PDF
            response = requests.get(pdf_url, headers=headers, stream=True, timeout=30)
            response.raise_for_status()
            
            # 保存PDF文件
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            print(f"下载成功: {file_name}")
            success_count += 1
            success_files.append(file_path)
            
            # 添加延迟，避免请求过于频繁
            time.sleep(0.5)
            
        except Exception as e:
            print(f"第 {index+1} 行: 下载失败 - {str(e)}")
            failed_count += 1
    
    print(f"\n下载完成! 尝试下载: {download_count}, 成功: {success_count}, 失败: {failed_count}")
    return success_files

def download_all_pdfs(folder_path, output_base_dir="PDF文件", max_downloads_per_file=None):
    """
    从文件夹中的所有Excel文件下载PDF
    :param folder_path: 包含Excel文件的文件夹路径
    :param output_base_dir: PDF文件保存的基础目录
    :param max_downloads_per_file: 每个Excel文件的最大下载数量，None表示下载全部
    """
    # 获取文件夹中所有Excel文件
    excel_files = glob.glob(os.path.join(folder_path, "*.xlsx"))
    
    if not excel_files:
        print("未找到Excel文件")
        return
    
    print(f"找到 {len(excel_files)} 个Excel文件")
    
    total_success = 0
    total_failed = 0
    
    # 处理每个Excel文件
    for excel_file in excel_files:
        file_name = os.path.basename(excel_file)
        print(f"\n=== 处理文件: {file_name} ===")
        success_files = download_pdfs_from_excel(excel_file, output_base_dir, max_downloads_per_file)
        
        total_success += len(success_files)
        total_failed += (max_downloads_per_file if max_downloads_per_file else 1000) - len(success_files)
    
    print(f"\n=== 总计 ===")
    print(f"成功下载: {total_success} 个PDF文件")
    print(f"失败: {total_failed} 个PDF文件")

if __name__ == "__main__":
    # 设置文件夹路径
    folder_path = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录
    
    # 下载PDF文件
    print("=== 下载PDF文件 ===")
    # 设置每个Excel文件的最大下载数量，None表示下载全部
    # 为了测试，可以先设置一个较小的数字，例如5
    max_downloads = None  # 设置为None以下载全部
    download_all_pdfs(folder_path, max_downloads_per_file=max_downloads)