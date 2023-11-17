# 讀取文件
with open('lable.txt', 'r') as file:
    data = file.read()

# 將浮點數轉換為整數
data = [int(float(num)) for num in data.split(', ')]

# 保存回文件
with open('label_int.txt', 'w') as file:
    file.write(', '.join(map(str, data)))

