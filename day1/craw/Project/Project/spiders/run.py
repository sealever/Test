import scrapy
import os
import sys

from Project.items import ProjectItem

sys.path.append(r'G:\AI-study\class\day1\craw\Project')
path = os.getcwd()
print(path)


class ScrapySpider(scrapy.Spider):
    name = 'run'
    allowed_domains = ['ssr1.scrape.center']
    start_urls = []

    for page in range(1, 11):
        cur = f"https://ssr1.scrape.center/page/{page}"
        start_urls.append(cur)

    def parse(self, response):
        url_hrefs = response.xpath('//a[@class="name"]/@href')
        for href in url_hrefs:
            href = href.extract()
            url = f"https://ssr1.scrape.center{href}"
            print(url)
            request = scrapy.Request(url, callback=self.parse_detail)
            print(request)
            yield request

    def parse_detail(self, response):

        def _extract_xpath(_xpath):
            _values = response.xpath(_xpath)
            if _values is None or len(_values) == 0:
                return []
            else:
                return [_v.extract() for _v in _values]

        def _extract_name():
            _values = _extract_xpath('//div[contains(@class, "item")]/div[2]/a/h2//text()')
            if len(_values) >= 1:
                return _values[0]
            else:
                return None

        _name = _extract_name()

        def _extract_country_duration():
            _values = _extract_xpath('//div[contains(@class, "item")]/div[2]/div[2]/span//text()')
            if len(_values) == 3:
                return _values[0], _values[2]
            else:
                return None, None

        country, duration = _extract_country_duration()

        def _extract_directors():
            _director_names = _extract_xpath('//div[contains(@class, "directors")]/div/div/div/p//text()')
            _director_images = _extract_xpath('//div[contains(@class, "directors")]/div/div/div/img//@src')
            if len(_director_images) == len(_director_names):
                _values = list(zip(_director_names, _director_images))
                return _values
            else:
                return []

        directors = _extract_directors()

        def _extract_actors():
            _actor_names = _extract_xpath('//div[contains(@class, "actors")]/div/div/div/p[1]//text()')
            _character_names = _extract_xpath('//div[contains(@class, "actors")]/div/div/div/p[2]//text()')
            if len(_actor_names) == len(_character_names):
                _names = list(zip(_actor_names, _character_names))
                return _names
            else:
                return []

        actors = _extract_actors()

        item = ProjectItem()
        item['name'] = _name
        item['country'] = country
        item['duration'] = duration
        item['directors'] = directors
        item['actors'] = actors

        return item
