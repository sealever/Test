import re
from urllib.parse import urljoin
import requests
import logging
import json
import os
from lxml import etree

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def scrape_page(url):
    logging.info(f'开始获取{url}...')
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        else:
            logging.error(f'获取异常的status——code：{response.status_code}；url：{url}')
    except Exception:
        logging.error(f'服务器异常，url：{url}', exc_info=True)
        return None


def scrape_index(page):
    url = f'https://ssr1.scrape.center/page/{page}'
    return scrape_page(url)


def parse_index(html_text):
    title1 = re.compile('<a.*?href="(.*?)".*?class="name"')
    items = re.findall(title1, html_text)
    if items is None or len(items) == 0:
        return None
    for item in items:
        detail_url = urljoin('https://ssr1.scrape.center', item)
        logging.info(f"获取得到详情页的url：{detail_url}")
        yield detail_url


def scrape_detail(url):
    return scrape_page(url)


def parse_detail(html_text):
    html = etree.HTML(html_text)
    name = html.xpath('//div[contains(@class, "item")]/div[2]/a/h2/text()')[0]
    times = html.xpath('//div[contains(@class, "item")]/div[2]/div[2]/span//text()')
    time = ''.join(times)
    intros = html.xpath('//div[contains(@class, "item")]/div[2]/div[4]//text()')
    intros = [i.strip('\n') for i in intros]
    intro = ''.join(intros)
    img = html.xpath('//div[contains(@class, "item")]/div[1]/a/img//@src')
    return {
        'name': name,
        'time': time,
        'intro': intro,
        'img': img
    }


def t0():
    index_text = scrape_index(page=1)
    detail_urls = parse_index(index_text)
    for detail_url in detail_urls:
        detail_text = scrape_detail(url=detail_url)
        data = parse_detail(detail_text)
        logging.info(f'解析结果为：{data}')


def save_data(result_output_path, data):
    name = data.get('name')
    name = name.split(" ")[0]
    data_path = rf"{result_output_path}/{name}.json"
    with open(data_path, "w", encoding="utf-8") as reader:
        json.dump(data, reader, ensure_ascii=False, indent=2)


def t2(result_output_path):
    os.makedirs(result_output_path, exist_ok=True)
    for page in range(1, 11):
        index_html_text = scrape_index(page=page)
        detail_urls = parse_index(index_html_text)
        for detail_url in detail_urls:
            detail_html_text = scrape_detail(url=detail_url)
            data = parse_detail(detail_html_text)
            save_data(result_output_path, data)


if __name__ == '__main__':
    t2(r'G:\AI-study\class\day1\craw\output')
