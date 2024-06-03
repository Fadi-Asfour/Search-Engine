from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule

from load_list import getList

from scrapy.spiders import CrawlSpider

class ExampleSpider(CrawlSpider):
    name = 'example'

    start_urls = getList()

    rules = (
        Rule(LinkExtractor(allow=()), callback='parse_item', follow=True),
    )

    custom_settings = {
        'CLOSESPIDER_PAGECOUNT': 1,
        'RETRY_TIMES': 5,
        'DOWNLOAD_TIMEOUT': 15,
        'ROBOTSTXT_OBEY': False,
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    }

    def parse_start_url(self, response):
        return self.parse_item(response)

    def parse_item(self, response):
        page_title = response.xpath('//title/text()').get()

        print("title:")
        print(page_title)

        yield {
                'title': page_title,
                'url': response.url,
            }

    def is_relevant(self, content):
        return True