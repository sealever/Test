import re
import requests


def t0():
    url = 'https://baike.baidu.com/wikicategory/view?categoryName=%E6%81%90%E9%BE%99%E5%A4%A7%E5%85%A8'
    response = requests.get(url, timeout=3)
    if response.status_code == 200:
        text = response.text
        print(text)


if __name__ == '__main__':
    t0()
