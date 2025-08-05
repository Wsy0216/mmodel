import pandas as pd

#  atheletes 加载 CSV 文件

# df = pd.read_csv(r"D:\pythonProject\pythonProject1\mcmdata\summerOly_athletes.csv")
# print(df.head())  # 打印前5行数据
# print(df.info())  # 查看数据结构
# print(df["Medal"].value_counts())  # 查看奖牌分布
# print(df.describe())  # 查看数值列的统计信息
# print(df.isnull().sum())  # 检查每列的缺失值数量
# print(f"重复行数：{df.duplicated().sum()}")  # 检查重复行
# df.drop_duplicates(inplace=True)  # 删除重复行
# df['Name'] = df['Name'].str.strip().str.title()  # 去空格并统一为首字母大写
# df['Sex'] = df['Sex'].str.strip().str.upper()   # 去空格并统一为大写
# df['Team'] = df['Team'].str.strip().str.title()
# df['NOC'] = df['NOC'].str.strip().str.upper()
# df['City'] = df['City'].str.strip().str.title()
# df['Sport'] = df['Sport'].str.strip().str.title()
# df['Event'] = df['Event'].str.strip()
# df['Team'] = df['Team'].str.split('/').str[0]
# print(df['Year'].describe())
# print(df['Sex'].unique())  # 查看唯一值
# print(df['Medal'].unique())
# df['Medal'] = df['Medal'].str.strip().str.title()  # 统一格式
# df = df.drop(columns=['City'])
# noc_to_country = {'CHN': 'China', 'DEN': 'Denmark'}  # 示例
# df['Country'] = df['NOC'].map(noc_to_country)
# df['Won_Medal'] = df['Medal'] != 'No medal'
# df.to_csv('cleaned_athletes_data.csv', index=False)

# hosts CSV

# df = pd.read_csv(r"D:\pythonProject\pythonProject1\mcmdata\summerOly_hosts.csv")
# print(df.head())  # 打印前5行数据
# print(df.info())  # 查看数据结构
# print(df.duplicated().sum())  # 检查重复行数量
# df = df.drop_duplicates()  # 删除重复行
# df['Host'] = df['Host'].str.strip().str.title()  # 去掉多余空格并统一首字母大写
# df[['City', 'Country']] = df['Host'].str.split(',', expand=True)
# df['Cancelled'] = df['Host'].str.contains('Cancelled', case=False)
# df['Country'] = df['Country'].str.strip()  # 去掉国家名多余的空格
# df = df[~df['Cancelled']]  # 过滤取消的年份
# df.to_csv('cleaned_hosts_data.csv', index=False)

# #  twice cleaning_hosts

# # 加载 CSV 文件
# df = pd.read_csv(r"D:\pythonProject\pythonProject1\mcmdata\cleaned_hosts_data.csv")
#
# # 重命名 'Country' 列为 'HostCountry'
# df = df.rename(columns={'Country': 'HostCountry'})
#
# # 删除 'Host'、'City' 和 'Cancelled' 列
# df = df.drop(columns=['Host', 'City', 'Cancelled'])
#
# # 打印清洗后的数据
# print(df.head())
#
# # 保存为新文件
# df.to_csv('twice_cleaned_hosts_data.csv', index=False)
# print("处理完成，保存为 final_cleaned_hosts_data.csv")

# medal_count

# df = pd.read_csv(r"D:\pythonProject\pythonProject1\mcmdata\summerOly_medal_counts.csv")
# print(df.info())  # 查看数据类型和缺失值
# print(df.isnull().sum())  # 检查每列缺失值数量
# print(df.duplicated().sum())  # 检查重复行数量

# programes

# df = pd.read_csv(r"D:\pythonProject\pythonProject1\mcmdata\summerOly_programs.csv", encoding='Windows-1252')
# print(df.info())  # 查看数据类型和缺失值情况
# print(df.head())  # 查看前5行数据
# df = df.drop(columns=[ 'Sports Governing Body'])  # 删除指定列
# df.columns = [col.strip('*') if col.isdigit() else col for col in df.columns]
# for col in df.columns[4:]:  # 假设年份列从第5列开始
#     df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
# # 按 Sport 聚合
# sport_totals = df.groupby('Sport').sum()
# print(sport_totals)
# df.to_csv("cleaned_programs_data.csv", index=False)

# import chardet
#
# # 检测文件编码
# with open(r"D:\pythonProject\pythonProject1\mcmdata\summerOly_programs.csv", 'rb') as f:
#     result = chardet.detect(f.read())
#
# print("检测到的文件编码:", result['encoding'])
#
# # 使用检测到的编码读取文件
# df = pd.read_csv(r"D:\pythonProject\pythonProject1\mcmdata\summerOly_programs.csv", encoding=result['encoding'])

# # atheletes NOC
# # 读取文件
# df = pd.read_csv(r"D:\pythonProject\pythonProject1\mcmdata\cleaned_athletes_data.csv")
#
# # 创建 NOC 映射表
# noc_to_country = {
#     'AFG': 'Afghanistan',
#     'ALB': 'Albania',
#     'ALG': 'Algeria',
#     'AND': 'Andorra',
#     'ANG': 'Angola',
#     'ANT': 'Antigua and Barbuda',
#     'ARU': 'Aruba',
#     'ASA': 'American Samoa',
#     'ARG': 'Argentina',
#     'ARM': 'Armenia',
#     'AUS': 'Australia',
#     'AUT': 'Austria',
#     'AZE': 'Azerbaijan',
#     'BAH': 'Bahamas',
#     'BAN': 'Bangladesh',
#     'BAR': 'Barbados',
#     'BDI': 'Burundi',
#     'BEL': 'Belgium',
#     'BEN': 'Benin',
#     'BER': 'Bermuda',
#     'BHU': 'Bhutan',
#     'BIH': 'Bosnia and Herzegovina',
#     'BIZ': 'Belize',
#     'BLR': 'Belarus',
#     'BOL': 'Bolivia',
#     'BOT': 'Botswana',
#     'BRA': 'Brazil',
#     'BRN': 'Bahrain',
#     'BRU': 'Brunei Darussalam',
#     'BUL': 'Bulgaria',
#     'BUR': 'Burkina Faso',
#     'CAF': 'Central African Republic',
#     'CAN': 'Canada',
#     'CAY': 'Cayman Islands',
#     'CGO': 'Republic of the Congo',
#     'CHA': 'Chad',
#     'CHI': 'Chile',
#     'CHN': 'China',
#     'CIV': 'Côte d\'Ivoire',
#     'CMR': 'Cameroon',
#     'COD': 'Democratic Republic of the Congo',
#     'COK': 'Cook Islands',
#     'COL': 'Colombia',
#     'COM': 'Comoros',
#     'CPV': 'Cabo Verde',
#     'CRC': 'Costa Rica',
#     'CRO': 'Croatia',
#     'CUB': 'Cuba',
#     'CYP': 'Cyprus',
#     'CZE': 'Czech Republic',
#     'DEN': 'Denmark',
#     'DJI': 'Djibouti',
#     'DMA': 'Dominica',
#     'DOM': 'Dominican Republic',
#     'ECU': 'Ecuador',
#     'EGY': 'Egypt',
#     'ERI': 'Eritrea',
#     'ESA': 'El Salvador',
#     'ESP': 'Spain',
#     'EST': 'Estonia',
#     'ETH': 'Ethiopia',
#     'FIJ': 'Fiji',
#     'FIN': 'Finland',
#     'FRA': 'France',
#     'FSM': 'Federated States of Micronesia',
#     'GAB': 'Gabon',
#     'GAM': 'Gambia',
#     'GBR': 'Great Britain',
#     'GBS': 'Guinea-Bissau',
#     'GEO': 'Georgia',
#     'GEQ': 'Equatorial Guinea',
#     'GER': 'Germany',
#     'GHA': 'Ghana',
#     'GRE': 'Greece',
#     'GRN': 'Grenada',
#     'GUA': 'Guatemala',
#     'GUI': 'Guinea',
#     'GUM': 'Guam',
#     'GUY': 'Guyana',
#     'HAI': 'Haiti',
#     'HKG': 'Hong Kong, China',
#     'HON': 'Honduras',
#     'HUN': 'Hungary',
#     'INA': 'Indonesia',
#     'IND': 'India',
#     'IRI': 'Islamic Republic of Iran',
#     'IRL': 'Ireland',
#     'IRQ': 'Iraq',
#     'ISL': 'Iceland',
#     'ISR': 'Israel',
#     'ISV': 'Virgin Islands (US)',
#     'ITA': 'Italy',
#     'IVB': 'Virgin Islands (British)',
#     'JAM': 'Jamaica',
#     'JOR': 'Jordan',
#     'JPN': 'Japan',
#     'KAZ': 'Kazakhstan',
#     'KEN': 'Kenya',
#     'KGZ': 'Kyrgyzstan',
#     'KIR': 'Kiribati',
#     'KOR': 'Republic of Korea',
#     'KOS': 'Kosovo',
#     'KSA': 'Saudi Arabia',
#     'KUW': 'Kuwait',
#     'LAO': 'Lao People\'s Democratic Republic',
#     'LAT': 'Latvia',
#     'LBN': 'Lebanon',
#     'LBR': 'Liberia',
#     'LCA': 'Saint Lucia',
#     'LES': 'Lesotho',
#     'LIE': 'Liechtenstein',
#     'LTU': 'Lithuania',
#     'LUX': 'Luxembourg',
#     'MAD': 'Madagascar',
#     'MAR': 'Morocco',
#     'MAS': 'Malaysia',
#     'MAW': 'Malawi',
#     'MDA': 'Republic of Moldova',
#     'MDV': 'Maldives',
#     'MEX': 'Mexico',
#     'MGL': 'Mongolia',
#     'MHL': 'Marshall Islands',
#     'MKD': 'North Macedonia',
#     'MLI': 'Mali',
#     'MLT': 'Malta',
#     'MNE': 'Montenegro',
#     'MON': 'Monaco',
#     'MOZ': 'Mozambique',
#     'MRI': 'Mauritius',
#     'MTN': 'Mauritania',
#     'MYA': 'Myanmar',
#     'NAM': 'Namibia',
#     'NCA': 'Nicaragua',
#     'NED': 'Netherlands',
#     'NEP': 'Nepal',
#     'NGR': 'Nigeria',
#     'NIG': 'Niger',
#     'NMI': 'Northern Mariana Islands',
#     'NOR': 'Norway',
#     'NRU': 'Nauru',
#     'NZL': 'New Zealand',
#     'OMA': 'Oman',
#     'PAK': 'Pakistan',
#     'PAN': 'Panama',
#     'PAR': 'Paraguay',
#     'PER': 'Peru',
#     'PHI': 'Philippines',
#     'PLE': 'Palestine',
#     'PLW': 'Palau',
#     'PNG': 'Papua New Guinea',
#     'POL': 'Poland',
#     'POR': 'Portugal',
#     'PRK': 'Democratic People\'s Republic of Korea',
#     'PUR': 'Puerto Rico',
#     'QAT': 'Qatar',
#     'ROU': 'Romania',
#     'RSA': 'South Africa',
#     'RUS': 'Russian Federation',
#     'RWA': 'Rwanda',
#     'SAM': 'Samoa',
#     'SEN': 'Senegal',
#     'SEY': 'Seychelles',
#     'SGP': 'Singapore',
#     'SKN': 'Saint Kitts and Nevis',
#     'SLE': 'Sierra Leone',
#     'SLO': 'Slovenia',
#     'SMR': 'San Marino',
#     'SOL': 'Solomon Islands',
#     'SOM': 'Somalia',
#     'SRB': 'Serbia',
#     'SRI': 'Sri Lanka',
#     'SSD': 'South Sudan',
#     'STP': 'Sao Tome and Principe',
#     'SUD': 'Sudan',
#     'SUI': 'Switzerland',
#     'SUR': 'Suriname',
#     'SVK': 'Slovakia',
#     'SWE': 'Sweden',
#     'SWZ': 'Eswatini',
#     'SYR': 'Syrian Arab Republic',
#     'TAN': 'Tanzania',
#     'TGA': 'Tonga',
#     'THA': 'Thailand',
#     'TJK': 'Tajikistan',
#     'TKM': 'Turkmenistan',
#     'TLS': 'Timor-Leste',
#     'TOG': 'Togo',
#     'TPE': 'Chinese Taipei',
#     'TTO': 'Trinidad and Tobago',
#     'TUN': 'Tunisia',
#     'TUR': 'Turkey',
#     'TUV': 'Tuvalu',
#     'UAE': 'United Arab Emirates',
#     'UGA': 'Uganda',
#     'UKR': 'Ukraine',
#     'URU': 'Uruguay',
#     'USA': 'United States of America',
#     'UZB': 'Uzbekistan',
#     'VAN': 'Vanuatu',
#     'VEN': 'Venezuela',
#     'VIE': 'Vietnam',
#     'VIN': 'Saint Vincent and the Grenadines',
#     'YEM': 'Yemen',
#     'ZAM': 'Zambia',
#     'ZIM': 'Zimbabwe'
# }
#
# # 替换 NOC 列为扩展名
# df['NOC'] = df['NOC'].map(noc_to_country)
#
# # 检查替换后的数据
# print(df.head())
#
# # 保存更新后的文件
# df.to_csv("expanded_athletes_data.csv", index=False)
# print("NOC 列扩展为完整国家名称，保存为 expanded_athletes_data.csv")

# # 填充NOC
# import pandas as pd
#
# df = pd.read_csv(r"D:\pythonProject\pythonProject1\mcmdata\expanded_athletes_data.csv")
#
# # 检查 NOC 列的缺失值数量
# print("填充前 NOC 列缺失值数量:", df['NOC'].isnull().sum())
#
# # 用 Team 列对应值填充 NOC 列的缺失值
# df['NOC'] = df['NOC'].fillna(df['Team'])
#
# # 检查填充后的结果
# print("填充后 NOC 列缺失值数量:", df['NOC'].isnull().sum())
#
# # 保存结果到新文件
# df.to_csv("filled_NOC_data.csv", index=False)
# print("缺失值已填充并保存为 filled_NOC_data.csv")

# # 合并atheletes medal
# import pandas as pd
#
# # 读取两个表格
# df1 = pd.read_csv(r"D:\pythonProject\pythonProject1\mcmdata\filled_NOC_data.csv")  # 第一张表
# df2 = pd.read_csv(r"D:\pythonProject\pythonProject1\mcmdata\summerOly_medal_counts.csv")  # 第二张表
#
# # 打印表格信息
# print("表1信息：", df1.info())
# print("表2信息：", df2.info())
#
# # 合并两表
# merged_df = pd.merge(df1, df2, on="NOC", how="inner")  # 按 NOC 列合并
#
# # 检查合并结果
# print("合并后的数据：")
# print(merged_df.head())
#
# # 保存合并后的数据到新文件
# merged_df.to_csv("merged_data.csv", index=False)
# print("合并结果已保存为 merged_data.csv")

# import pandas as pd
#
# # 假设 merged_df 是你需要保存的数据
# merged_df = pd.read_csv(r"D:\pythonProject\pythonProject1\mcmdata\merged_data.csv")
# merged_df.to_parquet("merged_data.parquet", index=False)
# print("数据已保存为 Parquet 文件：merged_data.parquet")
# df = pd.read_parquet("merged_data.parquet")
# print(df.head())
#
# # 查看数据基本信息
# print(df.info())
# # 查看数据基本信息
# print(merged_df.info())  # 查看数据类型和内存占用
# print(merged_df.describe())  # 数值型列的统计信息
# print(merged_df.isnull().sum())  # 检查缺失值
#
# import pandas as pd
#
# # 加载两个 CSV 文件
# athletes_df = pd.read_csv(r"D:\pythonProject\pythonProject1\mcmdata\cleaned_athletes_data.csv")
# medal_counts_df = pd.read_csv(r"D:\pythonProject\pythonProject1\mcmdata\summerOly_medal_counts.csv")
# noc_mapping = {
#     'AFG': 'Afghanistan',
#     'ALB': 'Albania',
#     'ALG': 'Algeria',
#     'AND': 'Andorra',
#     'ANG': 'Angola',
#     'ANT': 'Antigua and Barbuda',
#     'ARU': 'Aruba',
#     'ASA': 'American Samoa',
#     'ARG': 'Argentina',
#     'ARM': 'Armenia',
#     'AUS': 'Australia',
#     'AUT': 'Austria',
#     'AZE': 'Azerbaijan',
#     'BAH': 'Bahamas',
#     'BAN': 'Bangladesh',
#     'BAR': 'Barbados',
#     'BDI': 'Burundi',
#     'BEL': 'Belgium',
#     'BEN': 'Benin',
#     'BER': 'Bermuda',
#     'BHU': 'Bhutan',
#     'BIH': 'Bosnia and Herzegovina',
#     'BIZ': 'Belize',
#     'BLR': 'Belarus',
#     'BOL': 'Bolivia',
#     'BOT': 'Botswana',
#     'BRA': 'Brazil',
#     'BRN': 'Bahrain',
#     'BRU': 'Brunei Darussalam',
#     'BUL': 'Bulgaria',
#     'BUR': 'Burkina Faso',
#     'CAF': 'Central African Republic',
#     'CAN': 'Canada',
#     'CAY': 'Cayman Islands',
#     'CGO': 'Republic of the Congo',
#     'CHA': 'Chad',
#     'CHI': 'Chile',
#     'CHN': 'People\'s Republic of China',
#     'CIV': 'Côte d\'Ivoire',
#     'CMR': 'Cameroon',
#     'COD': 'Democratic Republic of the Congo',
#     'COK': 'Cook Islands',
#     'COL': 'Colombia',
#     'COM': 'Comoros',
#     'CPV': 'Cabo Verde',
#     'CRC': 'Costa Rica',
#     'CRO': 'Croatia',
#     'CUB': 'Cuba',
#     'CYP': 'Cyprus',
#     'CZE': 'Czech Republic',
#     'DEN': 'Denmark',
#     'DJI': 'Djibouti',
#     'DMA': 'Dominica',
#     'DOM': 'Dominican Republic',
#     'ECU': 'Ecuador',
#     'EGY': 'Egypt',
#     'ERI': 'Eritrea',
#     'ESA': 'El Salvador',
#     'ESP': 'Spain',
#     'EST': 'Estonia',
#     'ETH': 'Ethiopia',
#     'FIJ': 'Fiji',
#     'FIN': 'Finland',
#     'FRA': 'France',
#     'FSM': 'Federated States of Micronesia',
#     'GAB': 'Gabon',
#     'GAM': 'Gambia',
#     'GBR': 'Great Britain',
#     'GBS': 'Guinea-Bissau',
#     'GEO': 'Georgia',
#     'GEQ': 'Equatorial Guinea',
#     'GER': 'Germany',
#     'GHA': 'Ghana',
#     'GRE': 'Greece',
#     'GRN': 'Grenada',
#     'GUA': 'Guatemala',
#     'GUI': 'Guinea',
#     'GUM': 'Guam',
#     'GUY': 'Guyana',
#     'HAI': 'Haiti',
#     'HKG': 'Hong Kong, China',
#     'HON': 'Honduras',
#     'HUN': 'Hungary',
#     'INA': 'Indonesia',
#     'IND': 'India',
#     'IRI': 'Islamic Republic of Iran',
#     'IRL': 'Ireland',
#     'IRQ': 'Iraq',
#     'ISL': 'Iceland',
#     'ISR': 'Israel',
#     'ISV': 'Virgin Islands (US)',
#     'ITA': 'Italy',
#     'IVB': 'Virgin Islands (British)',
#     'JAM': 'Jamaica',
#     'JOR': 'Jordan',
#     'JPN': 'Japan',
#     'KAZ': 'Kazakhstan',
#     'KEN': 'Kenya',
#     'KGZ': 'Kyrgyzstan',
#     'KIR': 'Kiribati',
#     'KOR': 'Republic of Korea',
#     'KOS': 'Kosovo',
#     'KSA': 'Saudi Arabia',
#     'KUW': 'Kuwait',
#     'LAO': 'Lao People\'s Democratic Republic',
#     'LAT': 'Latvia',
#     'LBN': 'Lebanon',
#     'LBR': 'Liberia',
#     'LCA': 'Saint Lucia',
#     'LES': 'Lesotho',
#     'LIE': 'Liechtenstein',
#     'LTU': 'Lithuania',
#     'LUX': 'Luxembourg',
#     'MAD': 'Madagascar',
#     'MAR': 'Morocco',
#     'MAS': 'Malaysia',
#     'MAW': 'Malawi',
#     'MDA': 'Republic of Moldova',
#     'MDV': 'Maldives',
#     'MEX': 'Mexico',
#     'MGL': 'Mongolia',
#     'MHL': 'Marshall Islands',
#     'MKD': 'North Macedonia',
#     'MLI': 'Mali',
#     'MLT': 'Malta',
#     'MNE': 'Montenegro',
#     'MON': 'Monaco',
#     'MOZ': 'Mozambique',
#     'MRI': 'Mauritius',
#     'MTN': 'Mauritania',
#     'MYA': 'Myanmar',
#     'NAM': 'Namibia',
#     'NCA': 'Nicaragua',
#     'NED': 'Netherlands',
#     'NEP': 'Nepal',
#     'NGR': 'Nigeria',
#     'NIG': 'Niger',
#     'NMI': 'Northern Mariana Islands',
#     'NOR': 'Norway',
#     'NRU': 'Nauru',
#     'NZL': 'New Zealand',
#     'OMA': 'Oman',
#     'PAK': 'Pakistan',
#     'PAN': 'Panama',
#     'PAR': 'Paraguay',
#     'PER': 'Peru',
#     'PHI': 'Philippines',
#     'PLE': 'Palestine',
#     'PLW': 'Palau',
#     'PNG': 'Papua New Guinea',
#     'POL': 'Poland',
#     'POR': 'Portugal',
#     'PRK': 'Democratic People\'s Republic of Korea',
#     'PUR': 'Puerto Rico',
#     'QAT': 'Qatar',
#     'ROU': 'Romania',
#     'RSA': 'South Africa',
#     'RUS': 'Russian Federation',
#     'RWA': 'Rwanda',
#     'SAM': 'Samoa',
#     'SEN': 'Senegal',
#     'SEY': 'Seychelles',
#     'SGP': 'Singapore',
#     'SKN': 'Saint Kitts and Nevis',
#     'SLE': 'Sierra Leone',
#     'SLO': 'Slovenia',
#     'SMR': 'San Marino',
#     'SOL': 'Solomon Islands',
#     'SOM': 'Somalia',
#     'SRB': 'Serbia',
#     'SRI': 'Sri Lanka',
#     'SSD': 'South Sudan',
#     'STP': 'Sao Tome and Principe',
#     'SUD': 'Sudan',
#     'SUI': 'Switzerland',
#     'SUR': 'Suriname',
#     'SVK': 'Slovakia',
#     'SWE': 'Sweden',
#     'SWZ': 'Eswatini',
#     'SYR': 'Syrian Arab Republic',
#     'TAN': 'Tanzania',
#     'TGA': 'Tonga',
#     'THA': 'Thailand',
#     'TJK': 'Tajikistan',
#     'TKM': 'Turkmenistan',
#     'TLS': 'Timor-Leste',
#     'TOG': 'Togo',
#     'TPE': 'Chinese Taipei',
#     'TTO': 'Trinidad and Tobago',
#     'TUN': 'Tunisia',
#     'TUR': 'Turkey',
#     'TUV': 'Tuvalu',
#     'UAE': 'United Arab Emirates',
#     'UGA': 'Uganda',
#     'UKR': 'Ukraine',
#     'URU': 'Uruguay',
#     'USA': 'United States of America',
#     'UZB': 'Uzbekistan',
#     'VAN': 'Vanuatu',
#     'VEN': 'Venezuela',
#     'VIE': 'Vietnam',
#     'VIN': 'Saint Vincent and the Grenadines',
#     'YEM': 'Yemen',
#     'ZAM': 'Zambia',
#     'ZIM': 'Zimbabwe'
# }
# country_to_noc = {v: k for k, v in noc_mapping.items()}
# def convert_country_to_noc(country_name):
#     return country_to_noc.get(country_name, country_name)  # 如果找不到匹配，返回原始名称
# medal_counts_df['NOC'] = medal_counts_df['NOC'].apply(convert_country_to_noc)
# # 打印更新后的数据框
# print(medal_counts_df.head())
#
# # 保存更新后的数据框
# medal_counts_df.to_csv('updated_medal_counts.csv', index=False)

# # wonmedal 01编码
# import pandas as pd
#
# # 读取数据
# df = pd.read_csv(r"D:\pythonProject\pythonProject1\mcmdata\cleaned_athletes_data.csv")
#
# # 使用条件编码：'No Medal' 编码为 0，其他奖牌编码为 1
# df['Won_Medal'] = df['Medal'].apply(lambda x: 0 if x == 'No Medal' else 1)
#
# #删除多余country列
# df = df.drop(columns = ['Country'])
#
# # 查看更新后的数据
# print(df.head())
#
# # 保存更新后的文件
# df.to_csv('updated_atheletes.csv', index=False)

# # 重视程度 横向
# import pandas as pd
#
# # 读取数据
# df = pd.read_csv(r"D:\pythonProject\pythonProject1\mcmdata\cleaned_programs_data.csv", encoding='cp1252')
#
# # 计算每一年的总场次
# years = df.columns[3:]  # 从第二列开始是年份数据
# df['Total'] = df[years].sum(axis=1)
#
# # 计算每个项目在每一年中的重视程度
# for year in years:
#     df[f'Importance_{year}'] = df[year] / df['Total']
#
# # 查看结果
# print(df[['Sport', 'Total'] + [f'Importance_{year}' for year in years]].head())
#
# # 保存结果
# df.to_csv('importance.csv', index=False)

#重视程度 纵向
# import pandas as pd
#
# # 载入数据
# df = pd.read_csv(r"D:\pythonProject\pythonProject1\mcmdata\cleaned_programs_data.csv", encoding='cp1252')
#
#
# countries_columns = df.columns[3:]
#
# # 获取D37行的每个国家的得分
# reference_value = df.loc[71]  # D37是36行，注意从0开始
#
# # 对每个国家列进行除法操作，并生成对应的.per列
# for country in countries_columns:
#     new_column_name = f"{country}.per"
#     df[new_column_name] = df[country] / reference_value[country]  # 按照D37行的值进行计算
#
# # 保存计算后的文件
# df.to_csv('percentage.csv', index=False)
#
# # 显示部分计算结果
# print(df.head())

# # 添加hostcountry到medal表
# import pandas as pd
#
# # 载入第一个表（medal counts）
# medals_df = pd.read_excel(r"D:\pythonProject\pythonProject1\mcmdata\副本updated_medal_counts.xlsx")
#
# # 载入第二个表（hosts）
# hosts_df = pd.read_csv(r"D:\pythonProject\pythonProject1\mcmdata\twice_cleaned_hosts_data.csv")
#
# # 合并两个表格，基于"Year"列匹配
# merged_df = pd.merge(medals_df, hosts_df[['Year', 'HostCountry']], on='Year', how='left')
#
# # 查看合并后的数据
# print(merged_df.head())
#
# # 保存结果
# merged_df.to_csv('merged_1.csv', index=False)
#

# #merged_1 host转NOC
# import pandas as pd
#
# # 载入数据
# df = pd.read_csv(r"D:\pythonProject\pythonProject1\mcmdata\merged_1.csv")
#
# # 定义国家名称和NOC代码的映射
# country_to_noc = {
#     "Greece": "GRE",
#     "France": "FRA",
#     "United States": "USA",
#     "United Kingdom": "GBR",
#     "Sweden": "SWE",
#     "Belgium": "BEL",
#     "Netherlands": "NED",
#     "Germany": "GER",
#     "Finland": "FIN",
#     "Australia": "AUS",
#     "Italy": "ITA",
#     "Japan": "JPN",
#     "Mexico": "MEX",
#     "West Germany": "GER",  # 目前的NOC是GER
#     "Canada": "CAN",
#     "Soviet Union": "URS",  # 已解体
#     "South Korea": "KOR",
#     "Spain": "ESP",
#     "China": "CHN",
#     "Brazil": "BRA"
# }
#
# # 假设你的数据表中的国家名称列是 'Country'
# # 使用map函数替换国家名称为对应的NOC代码
# df['Country_NOC'] = df['HostCountry'].map(country_to_noc)
#
# # 查看替换后的结果
# print(df.head())
#
# # 保存结果到新的CSV文件
# df.to_csv('merged_2.csv', index=False)


# #删除缺失值
# import pandas as pd
# df = pd.read_csv(r"D:\pythonProject\pythonProject1\mcmdata\merged.csv",encoding='cp1252')
#
# df.dropna(inplace=True)
#  #保存结果到新的CSV文件
# df.to_csv('merged_4.csv', index=False)

# # 回归分析
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
# from sklearn.utils import resample
# from sklearn.metrics import r2_score, mean_squared_error
# import matplotlib.pyplot as plt
#
# # 载入数据
# df = pd.read_csv(r"D:\pythonProject\pythonProject1\mcmdata\merged_4.csv", encoding='cp1252')
#
# # 独热编码分类变量
# categorical_cols = ['Sport', 'HostCountry', 'Discipline', 'NOC']
# encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')  # 设置 handle_unknown='ignore'
# encoded_features = encoder.fit_transform(df[categorical_cols])
# encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols))
#
# # 合并编码后的特征和数值型特征
# X = pd.concat([df[['Year']], encoded_df], axis=1)
# y = df['Total']
#
# # 数据分割
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # 创建多项式特征
# poly = PolynomialFeatures(degree=2, include_bias=False)
# X_train_poly = poly.fit_transform(X_train)
# X_test_poly = poly.transform(X_test)
#
# # 模型训练
# model = LinearRegression()
# model.fit(X_train_poly, y_train)
#
# # 预测2028年数据（示例：法国主办）
# X_2028_raw = pd.DataFrame({
#     'Sport': ['Swimming'],
#     'Discipline': ['100m Freestyle'],
#     'Year': [2028],
#     'HostCountry': ['United States'],
#     'NOC': ['USA']
# })
# encoded_2028 = encoder.transform(X_2028_raw[categorical_cols])
# encoded_2028_df = pd.DataFrame(encoded_2028, columns=encoder.get_feature_names_out(categorical_cols))
# X_2028 = pd.concat([X_2028_raw[['Year']], encoded_2028_df], axis=1)
# X_2028_poly = poly.transform(X_2028)
#
# # 预测
# y_pred_2028 = model.predict(X_2028_poly)
#
# # 计算置信区间（自举法）
# n_bootstraps = 1000
# bootstrap_preds = []
# for _ in range(n_bootstraps):
#     X_resampled, y_resampled = resample(X_train_poly, y_train)
#     model.fit(X_resampled, y_resampled)
#     pred = model.predict(X_2028_poly)
#     bootstrap_preds.append(pred[0])
#
# lower_bound = np.percentile(bootstrap_preds, 2.5)
# upper_bound = np.percentile(bootstrap_preds, 97.5)
#
# print(f"预测的奖牌数: {y_pred_2028[0]:.2f}, 95%置信区间: [{lower_bound:.2f}, {upper_bound:.2f}]")
#
# # 模型评估
# y_pred = model.predict(X_test_poly)
# print(f"R² Score: {r2_score(y_test, y_pred):.2f}")
# print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
#
# # 可视化
# plt.figure(figsize=(10, 6))
# plt.bar(['France'], [y_pred_2028[0]], color='skyblue')
# plt.errorbar(['France'], [y_pred_2028[0]], yerr=[[y_pred_2028[0] - lower_bound], [upper_bound - y_pred_2028[0]]],
#              fmt='o', color='black', capsize=5)
# plt.xlabel('Country')
# plt.ylabel('Predicted Medal Count')
# plt.title('Predicted Medal Count for France in 2028 Olympics')
# plt.savefig('france_2028_prediction.png')

#柱状图
import pandas as pd

#  atheletes 加载 CSV 文件

# df = pd.read_csv(r"D:\pythonProject\pythonProject1\mcmdata\summerOly_athletes.csv")
# print(df.head())  # 打印前5行数据
# print(df.info())  # 查看数据结构
# print(df["Medal"].value_counts())  # 查看奖牌分布
# print(df.describe())  # 查看数值列的统计信息
# print(df.isnull().sum())  # 检查每列的缺失值数量
# print(f"重复行数：{df.duplicated().sum()}")  # 检查重复行
# df.drop_duplicates(inplace=True)  # 删除重复行
# df['Name'] = df['Name'].str.strip().str.title()  # 去空格并统一为首字母大写
# df['Sex'] = df['Sex'].str.strip().str.upper()   # 去空格并统一为大写
# df['Team'] = df['Team'].str.strip().str.title()
# df['NOC'] = df['NOC'].str.strip().str.upper()
# df['City'] = df['City'].str.strip().str.title()
# df['Sport'] = df['Sport'].str.strip().str.title()
# df['Event'] = df['Event'].str.strip()
# df['Team'] = df['Team'].str.split('/').str[0]
# print(df['Year'].describe())
# print(df['Sex'].unique())  # 查看唯一值
# print(df['Medal'].unique())
# df['Medal'] = df['Medal'].str.strip().str.title()  # 统一格式
# df = df.drop(columns=['City'])
# noc_to_country = {'CHN': 'China', 'DEN': 'Denmark'}  # 示例
# df['Country'] = df['NOC'].map(noc_to_country)
# df['Won_Medal'] = df['Medal'] != 'No medal'
# df.to_csv('cleaned_athletes_data.csv', index=False)

# hosts CSV

# df = pd.read_csv(r"D:\pythonProject\pythonProject1\mcmdata\summerOly_hosts.csv")
# print(df.head())  # 打印前5行数据
# print(df.info())  # 查看数据结构
# print(df.duplicated().sum())  # 检查重复行数量
# df = df.drop_duplicates()  # 删除重复行
# df['Host'] = df['Host'].str.strip().str.title()  # 去掉多余空格并统一首字母大写
# df[['City', 'Country']] = df['Host'].str.split(',', expand=True)
# df['Cancelled'] = df['Host'].str.contains('Cancelled', case=False)
# df['Country'] = df['Country'].str.strip()  # 去掉国家名多余的空格
# df = df[~df['Cancelled']]  # 过滤取消的年份
# df.to_csv('cleaned_hosts_data.csv', index=False)

# #  twice cleaning_hosts

# # 加载 CSV 文件
# df = pd.read_csv(r"D:\pythonProject\pythonProject1\mcmdata\cleaned_hosts_data.csv")
#
# # 重命名 'Country' 列为 'HostCountry'
# df = df.rename(columns={'Country': 'HostCountry'})
#
# # 删除 'Host'、'City' 和 'Cancelled' 列
# df = df.drop(columns=['Host', 'City', 'Cancelled'])
#
# # 打印清洗后的数据
# print(df.head())
#
# # 保存为新文件
# df.to_csv('twice_cleaned_hosts_data.csv', index=False)
# print("处理完成，保存为 final_cleaned_hosts_data.csv")

# medal_count

# df = pd.read_csv(r"D:\pythonProject\pythonProject1\mcmdata\summerOly_medal_counts.csv")
# print(df.info())  # 查看数据类型和缺失值
# print(df.isnull().sum())  # 检查每列缺失值数量
# print(df.duplicated().sum())  # 检查重复行数量

# programes

# df = pd.read_csv(r"D:\pythonProject\pythonProject1\mcmdata\summerOly_programs.csv", encoding='Windows-1252')
# print(df.info())  # 查看数据类型和缺失值情况
# print(df.head())  # 查看前5行数据
# df = df.drop(columns=[ 'Sports Governing Body'])  # 删除指定列
# df.columns = [col.strip('*') if col.isdigit() else col for col in df.columns]
# for col in df.columns[4:]:  # 假设年份列从第5列开始
#     df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
# # 按 Sport 聚合
# sport_totals = df.groupby('Sport').sum()
# print(sport_totals)
# df.to_csv("cleaned_programs_data.csv", index=False)

# import chardet
#
# # 检测文件编码
# with open(r"D:\pythonProject\pythonProject1\mcmdata\summerOly_programs.csv", 'rb') as f:
#     result = chardet.detect(f.read())
#
# print("检测到的文件编码:", result['encoding'])
#
# # 使用检测到的编码读取文件
# df = pd.read_csv(r"D:\pythonProject\pythonProject1\mcmdata\summerOly_programs.csv", encoding=result['encoding'])

# # atheletes NOC
# # 读取文件
# df = pd.read_csv(r"D:\pythonProject\pythonProject1\mcmdata\cleaned_athletes_data.csv")
#
# # 创建 NOC 映射表
# noc_to_country = {
#     'AFG': 'Afghanistan',
#     'ALB': 'Albania',
#     'ALG': 'Algeria',
#     'AND': 'Andorra',
#     'ANG': 'Angola',
#     'ANT': 'Antigua and Barbuda',
#     'ARU': 'Aruba',
#     'ASA': 'American Samoa',
#     'ARG': 'Argentina',
#     'ARM': 'Armenia',
#     'AUS': 'Australia',
#     'AUT': 'Austria',
#     'AZE': 'Azerbaijan',
#     'BAH': 'Bahamas',
#     'BAN': 'Bangladesh',
#     'BAR': 'Barbados',
#     'BDI': 'Burundi',
#     'BEL': 'Belgium',
#     'BEN': 'Benin',
#     'BER': 'Bermuda',
#     'BHU': 'Bhutan',
#     'BIH': 'Bosnia and Herzegovina',
#     'BIZ': 'Belize',
#     'BLR': 'Belarus',
#     'BOL': 'Bolivia',
#     'BOT': 'Botswana',
#     'BRA': 'Brazil',
#     'BRN': 'Bahrain',
#     'BRU': 'Brunei Darussalam',
#     'BUL': 'Bulgaria',
#     'BUR': 'Burkina Faso',
#     'CAF': 'Central African Republic',
#     'CAN': 'Canada',
#     'CAY': 'Cayman Islands',
#     'CGO': 'Republic of the Congo',
#     'CHA': 'Chad',
#     'CHI': 'Chile',
#     'CHN': 'China',
#     'CIV': 'Côte d\'Ivoire',
#     'CMR': 'Cameroon',
#     'COD': 'Democratic Republic of the Congo',
#     'COK': 'Cook Islands',
#     'COL': 'Colombia',
#     'COM': 'Comoros',
#     'CPV': 'Cabo Verde',
#     'CRC': 'Costa Rica',
#     'CRO': 'Croatia',
#     'CUB': 'Cuba',
#     'CYP': 'Cyprus',
#     'CZE': 'Czech Republic',
#     'DEN': 'Denmark',
#     'DJI': 'Djibouti',
#     'DMA': 'Dominica',
#     'DOM': 'Dominican Republic',
#     'ECU': 'Ecuador',
#     'EGY': 'Egypt',
#     'ERI': 'Eritrea',
#     'ESA': 'El Salvador',
#     'ESP': 'Spain',
#     'EST': 'Estonia',
#     'ETH': 'Ethiopia',
#     'FIJ': 'Fiji',
#     'FIN': 'Finland',
#     'FRA': 'France',
#     'FSM': 'Federated States of Micronesia',
#     'GAB': 'Gabon',
#     'GAM': 'Gambia',
#     'GBR': 'Great Britain',
#     'GBS': 'Guinea-Bissau',
#     'GEO': 'Georgia',
#     'GEQ': 'Equatorial Guinea',
#     'GER': 'Germany',
#     'GHA': 'Ghana',
#     'GRE': 'Greece',
#     'GRN': 'Grenada',
#     'GUA': 'Guatemala',
#     'GUI': 'Guinea',
#     'GUM': 'Guam',
#     'GUY': 'Guyana',
#     'HAI': 'Haiti',
#     'HKG': 'Hong Kong, China',
#     'HON': 'Honduras',
#     'HUN': 'Hungary',
#     'INA': 'Indonesia',
#     'IND': 'India',
#     'IRI': 'Islamic Republic of Iran',
#     'IRL': 'Ireland',
#     'IRQ': 'Iraq',
#     'ISL': 'Iceland',
#     'ISR': 'Israel',
#     'ISV': 'Virgin Islands (US)',
#     'ITA': 'Italy',
#     'IVB': 'Virgin Islands (British)',
#     'JAM': 'Jamaica',
#     'JOR': 'Jordan',
#     'JPN': 'Japan',
#     'KAZ': 'Kazakhstan',
#     'KEN': 'Kenya',
#     'KGZ': 'Kyrgyzstan',
#     'KIR': 'Kiribati',
#     'KOR': 'Republic of Korea',
#     'KOS': 'Kosovo',
#     'KSA': 'Saudi Arabia',
#     'KUW': 'Kuwait',
#     'LAO': 'Lao People\'s Democratic Republic',
#     'LAT': 'Latvia',
#     'LBN': 'Lebanon',
#     'LBR': 'Liberia',
#     'LCA': 'Saint Lucia',
#     'LES': 'Lesotho',
#     'LIE': 'Liechtenstein',
#     'LTU': 'Lithuania',
#     'LUX': 'Luxembourg',
#     'MAD': 'Madagascar',
#     'MAR': 'Morocco',
#     'MAS': 'Malaysia',
#     'MAW': 'Malawi',
#     'MDA': 'Republic of Moldova',
#     'MDV': 'Maldives',
#     'MEX': 'Mexico',
#     'MGL': 'Mongolia',
#     'MHL': 'Marshall Islands',
#     'MKD': 'North Macedonia',
#     'MLI': 'Mali',
#     'MLT': 'Malta',
#     'MNE': 'Montenegro',
#     'MON': 'Monaco',
#     'MOZ': 'Mozambique',
#     'MRI': 'Mauritius',
#     'MTN': 'Mauritania',
#     'MYA': 'Myanmar',
#     'NAM': 'Namibia',
#     'NCA': 'Nicaragua',
#     'NED': 'Netherlands',
#     'NEP': 'Nepal',
#     'NGR': 'Nigeria',
#     'NIG': 'Niger',
#     'NMI': 'Northern Mariana Islands',
#     'NOR': 'Norway',
#     'NRU': 'Nauru',
#     'NZL': 'New Zealand',
#     'OMA': 'Oman',
#     'PAK': 'Pakistan',
#     'PAN': 'Panama',
#     'PAR': 'Paraguay',
#     'PER': 'Peru',
#     'PHI': 'Philippines',
#     'PLE': 'Palestine',
#     'PLW': 'Palau',
#     'PNG': 'Papua New Guinea',
#     'POL': 'Poland',
#     'POR': 'Portugal',
#     'PRK': 'Democratic People\'s Republic of Korea',
#     'PUR': 'Puerto Rico',
#     'QAT': 'Qatar',
#     'ROU': 'Romania',
#     'RSA': 'South Africa',
#     'RUS': 'Russian Federation',
#     'RWA': 'Rwanda',
#     'SAM': 'Samoa',
#     'SEN': 'Senegal',
#     'SEY': 'Seychelles',
#     'SGP': 'Singapore',
#     'SKN': 'Saint Kitts and Nevis',
#     'SLE': 'Sierra Leone',
#     'SLO': 'Slovenia',
#     'SMR': 'San Marino',
#     'SOL': 'Solomon Islands',
#     'SOM': 'Somalia',
#     'SRB': 'Serbia',
#     'SRI': 'Sri Lanka',
#     'SSD': 'South Sudan',
#     'STP': 'Sao Tome and Principe',
#     'SUD': 'Sudan',
#     'SUI': 'Switzerland',
#     'SUR': 'Suriname',
#     'SVK': 'Slovakia',
#     'SWE': 'Sweden',
#     'SWZ': 'Eswatini',
#     'SYR': 'Syrian Arab Republic',
#     'TAN': 'Tanzania',
#     'TGA': 'Tonga',
#     'THA': 'Thailand',
#     'TJK': 'Tajikistan',
#     'TKM': 'Turkmenistan',
#     'TLS': 'Timor-Leste',
#     'TOG': 'Togo',
#     'TPE': 'Chinese Taipei',
#     'TTO': 'Trinidad and Tobago',
#     'TUN': 'Tunisia',
#     'TUR': 'Turkey',
#     'TUV': 'Tuvalu',
#     'UAE': 'United Arab Emirates',
#     'UGA': 'Uganda',
#     'UKR': 'Ukraine',
#     'URU': 'Uruguay',
#     'USA': 'United States of America',
#     'UZB': 'Uzbekistan',
#     'VAN': 'Vanuatu',
#     'VEN': 'Venezuela',
#     'VIE': 'Vietnam',
#     'VIN': 'Saint Vincent and the Grenadines',
#     'YEM': 'Yemen',
#     'ZAM': 'Zambia',
#     'ZIM': 'Zimbabwe'
# }
#
# # 替换 NOC 列为扩展名
# df['NOC'] = df['NOC'].map(noc_to_country)
#
# # 检查替换后的数据
# print(df.head())
#
# # 保存更新后的文件
# df.to_csv("expanded_athletes_data.csv", index=False)
# print("NOC 列扩展为完整国家名称，保存为 expanded_athletes_data.csv")

# # 填充NOC
# import pandas as pd
#
# df = pd.read_csv(r"D:\pythonProject\pythonProject1\mcmdata\expanded_athletes_data.csv")
#
# # 检查 NOC 列的缺失值数量
# print("填充前 NOC 列缺失值数量:", df['NOC'].isnull().sum())
#
# # 用 Team 列对应值填充 NOC 列的缺失值
# df['NOC'] = df['NOC'].fillna(df['Team'])
#
# # 检查填充后的结果
# print("填充后 NOC 列缺失值数量:", df['NOC'].isnull().sum())
#
# # 保存结果到新文件
# df.to_csv("filled_NOC_data.csv", index=False)
# print("缺失值已填充并保存为 filled_NOC_data.csv")

# # 合并atheletes medal
# import pandas as pd
#
# # 读取两个表格
# df1 = pd.read_csv(r"D:\pythonProject\pythonProject1\mcmdata\filled_NOC_data.csv")  # 第一张表
# df2 = pd.read_csv(r"D:\pythonProject\pythonProject1\mcmdata\summerOly_medal_counts.csv")  # 第二张表
#
# # 打印表格信息
# print("表1信息：", df1.info())
# print("表2信息：", df2.info())
#
# # 合并两表
# merged_df = pd.merge(df1, df2, on="NOC", how="inner")  # 按 NOC 列合并
#
# # 检查合并结果
# print("合并后的数据：")
# print(merged_df.head())
#
# # 保存合并后的数据到新文件
# merged_df.to_csv("merged_data.csv", index=False)
# print("合并结果已保存为 merged_data.csv")

# import pandas as pd
#
# # 假设 merged_df 是你需要保存的数据
# merged_df = pd.read_csv(r"D:\pythonProject\pythonProject1\mcmdata\merged_data.csv")
# merged_df.to_parquet("merged_data.parquet", index=False)
# print("数据已保存为 Parquet 文件：merged_data.parquet")
# df = pd.read_parquet("merged_data.parquet")
# print(df.head())
#
# # 查看数据基本信息
# print(df.info())
# # 查看数据基本信息
# print(merged_df.info())  # 查看数据类型和内存占用
# print(merged_df.describe())  # 数值型列的统计信息
# print(merged_df.isnull().sum())  # 检查缺失值
#
# import pandas as pd
#
# # 加载两个 CSV 文件
# athletes_df = pd.read_csv(r"D:\pythonProject\pythonProject1\mcmdata\cleaned_athletes_data.csv")
# medal_counts_df = pd.read_csv(r"D:\pythonProject\pythonProject1\mcmdata\summerOly_medal_counts.csv")
# noc_mapping = {
#     'AFG': 'Afghanistan',
#     'ALB': 'Albania',
#     'ALG': 'Algeria',
#     'AND': 'Andorra',
#     'ANG': 'Angola',
#     'ANT': 'Antigua and Barbuda',
#     'ARU': 'Aruba',
#     'ASA': 'American Samoa',
#     'ARG': 'Argentina',
#     'ARM': 'Armenia',
#     'AUS': 'Australia',
#     'AUT': 'Austria',
#     'AZE': 'Azerbaijan',
#     'BAH': 'Bahamas',
#     'BAN': 'Bangladesh',
#     'BAR': 'Barbados',
#     'BDI': 'Burundi',
#     'BEL': 'Belgium',
#     'BEN': 'Benin',
#     'BER': 'Bermuda',
#     'BHU': 'Bhutan',
#     'BIH': 'Bosnia and Herzegovina',
#     'BIZ': 'Belize',
#     'BLR': 'Belarus',
#     'BOL': 'Bolivia',
#     'BOT': 'Botswana',
#     'BRA': 'Brazil',
#     'BRN': 'Bahrain',
#     'BRU': 'Brunei Darussalam',
#     'BUL': 'Bulgaria',
#     'BUR': 'Burkina Faso',
#     'CAF': 'Central African Republic',
#     'CAN': 'Canada',
#     'CAY': 'Cayman Islands',
#     'CGO': 'Republic of the Congo',
#     'CHA': 'Chad',
#     'CHI': 'Chile',
#     'CHN': 'People\'s Republic of China',
#     'CIV': 'Côte d\'Ivoire',
#     'CMR': 'Cameroon',
#     'COD': 'Democratic Republic of the Congo',
#     'COK': 'Cook Islands',
#     'COL': 'Colombia',
#     'COM': 'Comoros',
#     'CPV': 'Cabo Verde',
#     'CRC': 'Costa Rica',
#     'CRO': 'Croatia',
#     'CUB': 'Cuba',
#     'CYP': 'Cyprus',
#     'CZE': 'Czech Republic',
#     'DEN': 'Denmark',
#     'DJI': 'Djibouti',
#     'DMA': 'Dominica',
#     'DOM': 'Dominican Republic',
#     'ECU': 'Ecuador',
#     'EGY': 'Egypt',
#     'ERI': 'Eritrea',
#     'ESA': 'El Salvador',
#     'ESP': 'Spain',
#     'EST': 'Estonia',
#     'ETH': 'Ethiopia',
#     'FIJ': 'Fiji',
#     'FIN': 'Finland',
#     'FRA': 'France',
#     'FSM': 'Federated States of Micronesia',
#     'GAB': 'Gabon',
#     'GAM': 'Gambia',
#     'GBR': 'Great Britain',
#     'GBS': 'Guinea-Bissau',
#     'GEO': 'Georgia',
#     'GEQ': 'Equatorial Guinea',
#     'GER': 'Germany',
#     'GHA': 'Ghana',
#     'GRE': 'Greece',
#     'GRN': 'Grenada',
#     'GUA': 'Guatemala',
#     'GUI': 'Guinea',
#     'GUM': 'Guam',
#     'GUY': 'Guyana',
#     'HAI': 'Haiti',
#     'HKG': 'Hong Kong, China',
#     'HON': 'Honduras',
#     'HUN': 'Hungary',
#     'INA': 'Indonesia',
#     'IND': 'India',
#     'IRI': 'Islamic Republic of Iran',
#     'IRL': 'Ireland',
#     'IRQ': 'Iraq',
#     'ISL': 'Iceland',
#     'ISR': 'Israel',
#     'ISV': 'Virgin Islands (US)',
#     'ITA': 'Italy',
#     'IVB': 'Virgin Islands (British)',
#     'JAM': 'Jamaica',
#     'JOR': 'Jordan',
#     'JPN': 'Japan',
#     'KAZ': 'Kazakhstan',
#     'KEN': 'Kenya',
#     'KGZ': 'Kyrgyzstan',
#     'KIR': 'Kiribati',
#     'KOR': 'Republic of Korea',
#     'KOS': 'Kosovo',
#     'KSA': 'Saudi Arabia',
#     'KUW': 'Kuwait',
#     'LAO': 'Lao People\'s Democratic Republic',
#     'LAT': 'Latvia',
#     'LBN': 'Lebanon',
#     'LBR': 'Liberia',
#     'LCA': 'Saint Lucia',
#     'LES': 'Lesotho',
#     'LIE': 'Liechtenstein',
#     'LTU': 'Lithuania',
#     'LUX': 'Luxembourg',
#     'MAD': 'Madagascar',
#     'MAR': 'Morocco',
#     'MAS': 'Malaysia',
#     'MAW': 'Malawi',
#     'MDA': 'Republic of Moldova',
#     'MDV': 'Maldives',
#     'MEX': 'Mexico',
#     'MGL': 'Mongolia',
#     'MHL': 'Marshall Islands',
#     'MKD': 'North Macedonia',
#     'MLI': 'Mali',
#     'MLT': 'Malta',
#     'MNE': 'Montenegro',
#     'MON': 'Monaco',
#     'MOZ': 'Mozambique',
#     'MRI': 'Mauritius',
#     'MTN': 'Mauritania',
#     'MYA': 'Myanmar',
#     'NAM': 'Namibia',
#     'NCA': 'Nicaragua',
#     'NED': 'Netherlands',
#     'NEP': 'Nepal',
#     'NGR': 'Nigeria',
#     'NIG': 'Niger',
#     'NMI': 'Northern Mariana Islands',
#     'NOR': 'Norway',
#     'NRU': 'Nauru',
#     'NZL': 'New Zealand',
#     'OMA': 'Oman',
#     'PAK': 'Pakistan',
#     'PAN': 'Panama',
#     'PAR': 'Paraguay',
#     'PER': 'Peru',
#     'PHI': 'Philippines',
#     'PLE': 'Palestine',
#     'PLW': 'Palau',
#     'PNG': 'Papua New Guinea',
#     'POL': 'Poland',
#     'POR': 'Portugal',
#     'PRK': 'Democratic People\'s Republic of Korea',
#     'PUR': 'Puerto Rico',
#     'QAT': 'Qatar',
#     'ROU': 'Romania',
#     'RSA': 'South Africa',
#     'RUS': 'Russian Federation',
#     'RWA': 'Rwanda',
#     'SAM': 'Samoa',
#     'SEN': 'Senegal',
#     'SEY': 'Seychelles',
#     'SGP': 'Singapore',
#     'SKN': 'Saint Kitts and Nevis',
#     'SLE': 'Sierra Leone',
#     'SLO': 'Slovenia',
#     'SMR': 'San Marino',
#     'SOL': 'Solomon Islands',
#     'SOM': 'Somalia',
#     'SRB': 'Serbia',
#     'SRI': 'Sri Lanka',
#     'SSD': 'South Sudan',
#     'STP': 'Sao Tome and Principe',
#     'SUD': 'Sudan',
#     'SUI': 'Switzerland',
#     'SUR': 'Suriname',
#     'SVK': 'Slovakia',
#     'SWE': 'Sweden',
#     'SWZ': 'Eswatini',
#     'SYR': 'Syrian Arab Republic',
#     'TAN': 'Tanzania',
#     'TGA': 'Tonga',
#     'THA': 'Thailand',
#     'TJK': 'Tajikistan',
#     'TKM': 'Turkmenistan',
#     'TLS': 'Timor-Leste',
#     'TOG': 'Togo',
#     'TPE': 'Chinese Taipei',
#     'TTO': 'Trinidad and Tobago',
#     'TUN': 'Tunisia',
#     'TUR': 'Turkey',
#     'TUV': 'Tuvalu',
#     'UAE': 'United Arab Emirates',
#     'UGA': 'Uganda',
#     'UKR': 'Ukraine',
#     'URU': 'Uruguay',
#     'USA': 'United States of America',
#     'UZB': 'Uzbekistan',
#     'VAN': 'Vanuatu',
#     'VEN': 'Venezuela',
#     'VIE': 'Vietnam',
#     'VIN': 'Saint Vincent and the Grenadines',
#     'YEM': 'Yemen',
#     'ZAM': 'Zambia',
#     'ZIM': 'Zimbabwe'
# }
# country_to_noc = {v: k for k, v in noc_mapping.items()}
# def convert_country_to_noc(country_name):
#     return country_to_noc.get(country_name, country_name)  # 如果找不到匹配，返回原始名称
# medal_counts_df['NOC'] = medal_counts_df['NOC'].apply(convert_country_to_noc)
# # 打印更新后的数据框
# print(medal_counts_df.head())
#
# # 保存更新后的数据框
# medal_counts_df.to_csv('updated_medal_counts.csv', index=False)

# # wonmedal 01编码
# import pandas as pd
#
# # 读取数据
# df = pd.read_csv(r"D:\pythonProject\pythonProject1\mcmdata\cleaned_athletes_data.csv")
#
# # 使用条件编码：'No Medal' 编码为 0，其他奖牌编码为 1
# df['Won_Medal'] = df['Medal'].apply(lambda x: 0 if x == 'No Medal' else 1)
#
# #删除多余country列
# df = df.drop(columns = ['Country'])
#
# # 查看更新后的数据
# print(df.head())
#
# # 保存更新后的文件
# df.to_csv('updated_atheletes.csv', index=False)

# # 重视程度 横向
# import pandas as pd
#
# # 读取数据
# df = pd.read_csv(r"D:\pythonProject\pythonProject1\mcmdata\cleaned_programs_data.csv", encoding='cp1252')
#
# # 计算每一年的总场次
# years = df.columns[3:]  # 从第二列开始是年份数据
# df['Total'] = df[years].sum(axis=1)
#
# # 计算每个项目在每一年中的重视程度
# for year in years:
#     df[f'Importance_{year}'] = df[year] / df['Total']
#
# # 查看结果
# print(df[['Sport', 'Total'] + [f'Importance_{year}' for year in years]].head())
#
# # 保存结果
# df.to_csv('importance.csv', index=False)

#重视程度 纵向
# import pandas as pd
#
# # 载入数据
# df = pd.read_csv(r"D:\pythonProject\pythonProject1\mcmdata\cleaned_programs_data.csv", encoding='cp1252')
#
#
# countries_columns = df.columns[3:]
#
# # 获取D37行的每个国家的得分
# reference_value = df.loc[71]  # D37是36行，注意从0开始
#
# # 对每个国家列进行除法操作，并生成对应的.per列
# for country in countries_columns:
#     new_column_name = f"{country}.per"
#     df[new_column_name] = df[country] / reference_value[country]  # 按照D37行的值进行计算
#
# # 保存计算后的文件
# df.to_csv('percentage.csv', index=False)
#
# # 显示部分计算结果
# print(df.head())

# # 添加hostcountry到medal表
# import pandas as pd
#
# # 载入第一个表（medal counts）
# medals_df = pd.read_excel(r"D:\pythonProject\pythonProject1\mcmdata\副本updated_medal_counts.xlsx")
#
# # 载入第二个表（hosts）
# hosts_df = pd.read_csv(r"D:\pythonProject\pythonProject1\mcmdata\twice_cleaned_hosts_data.csv")
#
# # 合并两个表格，基于"Year"列匹配
# merged_df = pd.merge(medals_df, hosts_df[['Year', 'HostCountry']], on='Year', how='left')
#
# # 查看合并后的数据
# print(merged_df.head())
#
# # 保存结果
# merged_df.to_csv('merged_1.csv', index=False)
#

# #merged_1 host转NOC
# import pandas as pd
#
# # 载入数据
# df = pd.read_csv(r"D:\pythonProject\pythonProject1\mcmdata\merged_1.csv")
#
# # 定义国家名称和NOC代码的映射
# country_to_noc = {
#     "Greece": "GRE",
#     "France": "FRA",
#     "United States": "USA",
#     "United Kingdom": "GBR",
#     "Sweden": "SWE",
#     "Belgium": "BEL",
#     "Netherlands": "NED",
#     "Germany": "GER",
#     "Finland": "FIN",
#     "Australia": "AUS",
#     "Italy": "ITA",
#     "Japan": "JPN",
#     "Mexico": "MEX",
#     "West Germany": "GER",  # 目前的NOC是GER
#     "Canada": "CAN",
#     "Soviet Union": "URS",  # 已解体
#     "South Korea": "KOR",
#     "Spain": "ESP",
#     "China": "CHN",
#     "Brazil": "BRA"
# }
#
# # 假设你的数据表中的国家名称列是 'Country'
# # 使用map函数替换国家名称为对应的NOC代码
# df['Country_NOC'] = df['HostCountry'].map(country_to_noc)
#
# # 查看替换后的结果
# print(df.head())
#
# # 保存结果到新的CSV文件
# df.to_csv('merged_2.csv', index=False)


# #删除缺失值
# import pandas as pd
# df = pd.read_csv(r"D:\pythonProject\pythonProject1\mcmdata\merged.csv",encoding='cp1252')
#
# df.dropna(inplace=True)
#  #保存结果到新的CSV文件
# df.to_csv('merged_4.csv', index=False)

# 回归分析
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np

# 载入数据
df = pd.read_csv(r"D:\pythonProject\pythonProject1\mcmdata\merged_4.csv",encoding='cp1252')


# 标签编码
encoder = LabelEncoder()
df['Sport'] = encoder.fit_transform(df['Sport'])
df['HostCountry'] = encoder.fit_transform(df['HostCountry'])
df['Discipline'] = encoder.fit_transform(df['Discipline'])
df['NOC'] = encoder.fit_transform(df['NOC'])

# 选择特征和目标变量
X = df[['Sport', 'Discipline', 'Year', 'HostCountry', 'NOC']]  # 特征
y = df['Total']  # 目标变量（总奖牌数）

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 模型训练
model = LinearRegression()
model.fit(X_train, y_train)

# 预测2028年奥运奖牌数
X_2028 = pd.DataFrame([[0, 1, 2028, 0, 10]], columns=['Sport', 'Discipline', 'Year', 'HostCountry', 'NOC'])  # 假设输入数据
y_pred_2028 = model.predict(X_2028)

# 计算预测区间（例如95%的置信区间）
lower_bound = np.percentile(y_pred_2028, 2.5)
upper_bound = np.percentile(y_pred_2028, 97.5)

print(f"预测的奖牌数: {y_pred_2028}, 区间: [{lower_bound}, {upper_bound}]")
#可视化
import matplotlib
matplotlib.use('Agg')  # 或者 matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

# 示例数据
countries = ['USA', 'China', 'Germany', 'Australia', 'Japan']
predicted_medals = [32.5, 28.1, 25.7, 23.9, 20.3]
lower_bound = [30.0, 26.0, 23.0, 21.0, 18.0]
upper_bound = [35.0, 30.0, 28.0, 26.0, 23.0]

# 创建图表
plt.figure(figsize=(10, 6))

# 绘制每个国家的预测奖牌数及区间
plt.bar(countries, predicted_medals, color='skyblue', label='Predicted Medal Count')

# 绘制区间
plt.errorbar(countries, predicted_medals, yerr=[np.subtract(predicted_medals, lower_bound), np.subtract(upper_bound, predicted_medals)],
             fmt='o', color='black', label='Confidence Interval', capsize=5)

# 设置标签和标题
plt.xlabel('Countries')
plt.ylabel('Predicted Medal Count')
plt.title('Predicted Medal Counts with Confidence Intervals for 2028 Olympics')

# 添加图例
plt.legend()

# 显示图表
plt.show()