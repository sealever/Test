from day1.Flask.app import app
import os
import sys


# 将当前文件所在的文件夹添加到环境变量中
sys.path.append(os.path.dirname(__name__))
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=12233)