# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class ProjectItem(scrapy.Item):
    # define the fields for your item here like:
    name = scrapy.Field()
    country = scrapy.Field()
    duration = scrapy.Field()
    directors = scrapy.Field()
    actors = scrapy.Field()



