import requests
import urllib.parse as up
import time
import os

# 全局变量
major_url = 'https://image.baidu.com/search/index?'
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                         'AppleWebKit/537.36 (KHTML, like Gecko) '
                         'Chrome/84.0.4147.135 Safari/537.36'}
kws = ['工业机器人', '数控机床', '数控系统', '书籍', '图表', '人类']
data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
path_destination = os.path.join(data_root, "data_set", "industry_data", "industry_photos")


# 下载page=多少页数据，每页30张图片，需要自己更改page参数！
def pic_spider(kw, start_page=0, page=10, file_path=os.getcwd()):
    num_name = 0
    # path = os.path.join(file_path, kw)
    path = os.path.join(file_path, "temp_images")
    if not os.path.exists(path):
        os.mkdir(path)
    if kw != '':
        for num in range(page):
            # 设置起始页
            if num < start_page:
                num_name = num_name + 30
                continue
            # 比较几个请求传递参数后，只有 queryWord, word, pn, gsm 传递的参数变化
            # queryWord 和 word 都是要搜索的关键词
            # pn 图片数量
            # gsm 图片数量所对应的八进制
            data = {
                "tn": "resultjson_com",
                "logid": "11587207680030063767",
                "ipn": "rj",
                "ct": "201326592",
                "is": "",
                "fp": "result",
                "queryWord": kw,
                "cl": "2",
                "lm": "-1",
                "ie": "utf-8",
                "oe": "utf-8",
                "adpicid": "",
                "st": "-1",
                "z": "",
                "ic": "0",
                "hd": "",
                "latest": "",
                "copyright": "",
                "word": kw,
                "s": "",
                "se": "",
                "tab": "",
                "width": "",
                "height": "",
                "face": "0",
                "istype": "2",
                "qc": "",
                "nc": "1",
                "fr": "",
                "expermode": "",
                "force": "",
                "pn": num * 30,
                "rn": "30",
                "gsm": oct(num * 30),
                "1602481599433": ""
            }
            url = major_url + up.urlencode(data)
            i = 0
            pic_list = []
            while i < 5:
                try:
                    pic_list = requests.get(url=url, headers=headers).json().get('data')
                    break
                except:
                    print('网络不好，正在重试...')
                    i += 1
                    time.sleep(1.3)
            for pic in pic_list:
                url = pic.get('thumbURL', '')  # 有的没有图片链接，就设置成空
                if url == '':
                    continue
                # name = pic.get('fromPageTitleEnc')
                # for char in ['?', '\\', '/', '*', '"', '|', ':', '<', '>']:
                #     name = name.replace(char, '')   # 将所有不能出现在文件名中的字符去除掉
                name = kw + "_" + str(num_name)  # 下载时根据关键字和序号命名
                type = pic.get('type', 'png')  # 找到图片的类型，若没有找到，默认为 png
                pic_path = (os.path.join(path, '%s.%s') % (name, type))
                print(name, '已完成下载')
                if not os.path.exists(pic_path):
                    with open(pic_path, 'wb') as f:
                        f.write(requests.get(url=url, headers=headers).content)
                        num_name = num_name + 1


# 下载数据图片
def download():
    for i in range(len(kws)):
        # 使用爬虫下载图片
        pic_spider(kws[i], 0, 10)  # 自己设置多少页到多少页


# 根据下载的图片进行分类，并且保存到临时文件夹下
if __name__ == '__main__':
    download()
