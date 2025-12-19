"""
Author: Peter Li
Date: 2025-11-09
Description: 文件描述
"""

import os
import re

def get_company_codes_from_filenames(directory):
    """
    从指定目录中的PDF文件名提取公司代码
    :param directory: 包含PDF文件的目录路径
    :return: 公司代码集合
    """
    company_codes = set()
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            if filename.endswith('.pdf'):
                # 从文件名提取公司代码（文件名格式：公司代码_公司名称_年份.pdf）
                match = re.match(r'^(\d+)_', filename)
                if match:
                    company_codes.add(match.group(1))
    return company_codes

def find_companies_with_complete_annual_reports(base_directory, start_year, end_year):
    """
    查找在指定年份范围内每年都有年报的公司
    :param base_directory: 基础目录路径
    :param start_year: 开始年份
    :param end_year: 结束年份
    :return: 每年都有年报的公司代码列表
    """
    # 获取所有年份的公司代码集合
    all_company_codes = []
    
    for year in range(start_year, end_year + 1):
        year_directory = os.path.join(base_directory, f"{year}年年报")
        company_codes = get_company_codes_from_filenames(year_directory)
        all_company_codes.append(company_codes)
        print(f"{year}年年报公司数量: {len(company_codes)}")
    
    # 如果没有年份数据，返回空列表
    if not all_company_codes:
        return []
    
    # 计算所有年份公司代码的交集
    common_company_codes = set(all_company_codes[0])
    for company_codes in all_company_codes[1:]:
        common_company_codes = common_company_codes.intersection(company_codes)
    
    return list(common_company_codes)

def remove_incomplete_company_reports(base_directory, start_year, end_year, complete_companies):
    """
    删除年报不齐全的公司年报文件
    :param base_directory: 基础目录路径
    :param start_year: 开始年份
    :param end_year: 结束年份
    :param complete_companies: 每年都有年报的公司代码列表
    """
    # 将完整公司代码转换为集合以提高查找效率
    complete_company_set = set(complete_companies)
    
    # 统计删除的文件数量
    deleted_count = 0
    
    # 遍历年份目录
    for year in range(start_year, end_year + 1):
        year_directory = os.path.join(base_directory, f"{year}年年报")
        
        # 检查年份目录是否存在
        if not os.path.exists(year_directory):
            print(f"警告: {year_directory} 不存在")
            continue
            
        # 遍历目录中的所有PDF文件
        for filename in os.listdir(year_directory):
            if filename.endswith('.pdf'):
                # 从文件名提取公司代码
                match = re.match(r'^(\d+)_', filename)
                if match:
                    company_code = match.group(1)
                    # 如果公司代码不在完整公司列表中，则删除文件
                    if company_code not in complete_company_set:
                        file_path = os.path.join(year_directory, filename)
                        try:
                            os.remove(file_path)
                            print(f"已删除: {file_path}")
                            deleted_count += 1
                        except Exception as e:
                            print(f"删除文件失败 {file_path}: {e}")
    
    print(f"总共删除了 {deleted_count} 个不完整的年报文件")

def main():
    # 基础目录路径
    base_directory = r"H:\SRTP\程序\PDF文件\制造业"
    
    # 检查基础目录是否存在
    if not os.path.exists(base_directory):
        print(f"基础目录不存在: {base_directory}")
        return
    
    # 查找2018-2022年每年都有年报的公司
    complete_companies = find_companies_with_complete_annual_reports(base_directory, 2018, 2023)
    
    print(f"\n2018-2023年每年都有年报的公司数量: {len(complete_companies)}")
    print("公司代码列表:")
    for code in sorted(complete_companies):
        print(code)
    
    # 将结果保存到文件
    output_file = os.path.join(os.path.dirname(base_directory), "年报齐全的公司代码.txt")
    with open(output_file, 'a', encoding='utf-8') as f:
        for code in sorted(complete_companies):
            f.write(code + '\n')
    
    print(f"\n结果已保存到: {output_file}")
    
    # 删除年报不齐全的公司年报文件
    print("\n开始删除年报不齐全的公司年报文件...")
    remove_incomplete_company_reports(base_directory, 2018, 2023, complete_companies)
    print("删除操作完成")

if __name__ == "__main__":
    main()