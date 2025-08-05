import os

# 绝对路径
file_path = r"D:\pythonProject\pythonProject1\mcmdata\summerOly_athletes.csv"

# 检查文件是否存在
if os.path.exists(file_path):
    print(f"文件 {file_path} 存在！")
else:
    print(f"文件 {file_path} 不存在，请检查路径！")

