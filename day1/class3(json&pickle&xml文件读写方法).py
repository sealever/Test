from xml.etree import ElementTree as ET
import json  # json 为文本文件，可直接读写
import pickle  # pickle 为二进制文件，进行读写时需加‘b’,其余用法与json相同
"""'   json/pickle文件写入及读取方法   """
date = {'name': 'tom', 'age': 28, 123: 456}


with open('G:\AI-study\class\dir1\dir2\qwe.txt', mode='w+') as file:
    json.dump(date, file)  # json 序列化 ，将当前文件中的date转化为json式字符串写入qwe.txt文件中
    # 等效于：
    # str1 = json.dumps(date)
    # file.write(str1)
with open('G:\AI-study\class\dir1\dir2\qwe.txt') as file:
    res = json.load(file)
    # 等效于：
    # str1 = file.read()
    # res = json.loads(str1)
    print(res)


''''   xml文件内容读取方法   '''
tree = ET.parse('G:\AI-study\class\json&xml\json&xml\TestFile.xml')   # 对文件TestFile.xml进行解析
root = tree.getroot()   # 访问根元素
obj = root.find('./outputs/object')  # 查找至当前目录下的子目录outputs的子目录object
for i in obj:
    bdx = i.find('bndbox')  
    for j in bdx:
        print(j.text)
    print()
