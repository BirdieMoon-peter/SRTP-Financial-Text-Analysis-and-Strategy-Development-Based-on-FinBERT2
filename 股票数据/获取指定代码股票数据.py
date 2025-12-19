import pandas as pd
import time
import tushare as ts
import os
import re
import warnings
warnings.filterwarnings("ignore")

# 设置Tushare的token（需要先在Tushare官网注册获取token）
# 请将下面的'your_token_here'替换为你自己的Tushare token
ts.set_token('7f0f9072e0f93d39adcf4e1776a04351c69cb7fe7fb8800923fe20bd')
pro = ts.pro_api()

"""获取A股历史行情数据"""

all_prices = pd.DataFrame()
end_date = '20251109'
start_date = '20180101'

# 从"PDF文件/制造业/2018年年报"文件夹中动态读取股票代码
pdf_directory = "PDF文件/1"
stock_codes_set = set()
if os.path.isdir(pdf_directory):
    print(f"正在从 '{pdf_directory}' 文件夹中读取股票代码...")
    for filename in os.listdir(pdf_directory):
        if filename.endswith('.pdf'):
            match = re.match(r'(\d+)_.*\.pdf', filename)
            if match:
                stock_codes_set.add(match.group(1))
    stock_codes = sorted(list(stock_codes_set))
    print(f"成功读取 {len(stock_codes)} 个唯一的股票代码。")
else:
    print(f"目录 '{pdf_directory}' 不存在。请检查路径是否正确。")
    stock_codes = []

# 获取股票名称信息
stock_names = {}
try:
    # 获取所有股票基本信息
    stock_info = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name')
    for index, row in stock_info.iterrows():
        # 提取股票代码（去掉.SH或.SZ后缀）
        code = row['ts_code'].split('.')[0]
        stock_names[code] = row['name']
except Exception as e:
    print(f"获取股票名称信息时出错: {e}")

for code in stock_codes: 
    try:
        # 确保股票代码格式正确（沪市股票需要加.SH，深市股票需要加.SZ）
        if code.startswith('6'):
            ts_code = code + '.SH'
        else:
            ts_code = code + '.SZ'
        
        # 使用Tushare获取前复权历史行情数据
        # Tushare的日期格式为YYYYMMDD，与原代码一致
        stock_hist = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        time.sleep(1.5)  # Tushare API有访问频率限制，需要适当延时
        
        # 重命名列以匹配原代码
        stock_hist = stock_hist.rename(columns={'trade_date': '日期', 'close': '收盘'})
        
        # 我们只保留日期和收盘价
        stock_hist = stock_hist[['日期', '收盘']]
        stock_hist['stock_code'] = code
        # 添加股票名称列
        stock_hist['stock_name'] = stock_names.get(code, 'Unknown')
        all_prices = pd.concat([all_prices, stock_hist])
        print(f"成功获取股票 {code} ({stock_names.get(code, 'Unknown')}) 的数据。")
    except Exception as e:
        print(f"获取股票 {code} 数据时出错: {e}")
        continue

# 数据透视处理
pivot_prices = all_prices.pivot(index='日期', columns='stock_code', values='收盘')
pivot_prices.index = pd.to_datetime(pivot_prices.index)

# 对列进行排序
pivot_prices = pivot_prices.sort_index(axis=1)

# 创建一个新的DataFrame用于存储股票信息
stock_info_df = pd.DataFrame({
    '股票代码': pivot_prices.columns,
    '股票名称': [stock_names.get(str(code), 'Unknown') for code in pivot_prices.columns]
})

# 重置索引，使日期成为一列
pivot_prices_reset = pivot_prices.reset_index()

# 创建Excel写入器
output_excel_path = "补.xlsx"
with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
    # 将数据透视表写入"历史价格数据"工作表
    pivot_prices_reset.to_excel(writer, sheet_name='历史价格数据', index=False)
    
    # 将股票信息写入"股票信息"工作表
    stock_info_df.to_excel(writer, sheet_name='股票信息', index=False)

print(f"\n数据已成功保存到文件: {output_excel_path}")
print("Excel文件包含两个工作表:")
print("1. '历史价格数据' - 包含日期和各股票的历史收盘价")
print("2. '股票信息' - 包含股票代码和对应的股票名称")