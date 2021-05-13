import requests
import urllib.parse as up
import time
import os
import json
import torch
from PIL import Image
from torchvision import transforms
from matplotlib import font_manager as fm, rcParams
import matplotlib.pyplot as plt
from model import vgg


major_url = 'https://image.baidu.com/search/index?'
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                         'AppleWebKit/537.36 (KHTML, like Gecko) '
                         'Chrome/84.0.4147.135 Safari/537.36'}
kws = ['工业机器人', '数控机床', '数控系统', '书籍', '图表', '人类']
data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
path_destination = os.path.join(data_root, "data_set", "industry_data", "industry_photos")
num_class = 6


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


# 分类图片
def predictImage(img="../1.jpeg"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # load image
    img_path = img
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    # plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model， 记得修改num_classes=类别数
    model = vgg(model_name="vgg16", num_classes=num_class).to(device)
    # load model weights
    weights_path = "./vgg16Net.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
    class_fication = class_indict[str(predict_cla)]
    # print(class_fication)
    prob = predict[predict_cla].numpy()
    # print(prob)
    print_res = "class: {}   prob: {:.3}".format(class_fication, prob)
    if prob > 0.9:
        paths = img_path.split('\\')
        lens = len(paths)
        print(paths[lens - 1] + "  的图片分类结果:  " + print_res)
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False  # 这两行需要手动设置
    # plt.title(print_res)
    # plt.show()
    return class_fication, prob


def download():
    # 下载数据图片
    for i in range(len(kws)):
        # 使用爬虫下载图片
        pic_spider(kws[i], 0, 5)  # 自己设置多少页到多少页
        # 一边下载一边保存
        # 图片要保存的目的目录
        # image_destination = os.path.join(path_destination, kws[i])
        # # print(image_destination)
        # if not os.path.exists(image_destination):
        #     os.mkdir(image_destination)
        # # 对图片进行分类
        # image_path = os.path.join(os.path.join(os.getcwd()), kws[i])
        # img_list = os.listdir(image_path)
        # # 按照序号排序
        # img_list.sort(key=lambda x: int(x.split('.')[0]))
        # # count = len(img_list)
        # # print(count)
        # for img in img_list:
        #     image_path_name = os.path.join(image_path, img)
        #     if predictImage(image_path_name) > 0.9:
        #         image_open = Image.open(image_path_name)
        #         image_open_destination = os.path.join(image_destination, str(img))
        #         print(image_open_destination)
        #         image_open.save(image_open_destination, 'png')


def classfication():
    # 图片临时目录
    image_path = os.path.join(os.path.join(os.getcwd()), "temp_images")
    # 获取图片列表（图片名.后缀名）
    img_list = os.listdir(image_path)
    # 按照序号排序
    img_list.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    count = len(img_list)
    print("-------------总计{}张图片, 分类开始--------------".format(count))
    # 遍历图片列表
    for img in img_list:
        # 图片临时保存全路径
        image_path_name = os.path.join(image_path, img)
        try:
            post_class_fication, prob = predictImage(image_path_name)
            if prob > 0.9:
                # 加载图片
                image_open = Image.open(image_path_name)
                # 图片目的保存目标
                image_dir_destination = os.path.join(path_destination, str(post_class_fication))
                if not os.path.exists(image_dir_destination):
                    os.mkdir(image_dir_destination)
                # 图片目的保存文件名全路径
                image_save_destination = os.path.join(image_dir_destination, str(img))
                print("分类结果图片保存路径为: " + image_save_destination)
                # 保存图片
                image_open.save(image_save_destination, 'png')
        except Exception as e:
            print(e)
            continue

    print("-------------总计{}张图片, 分类结束--------------".format(count))


# 根据下载的图片进行分类，并且保存到数据集目录下
if __name__ == '__main__':
    # 下载数据图片
    # download()
    # 对图片进行分类
    classfication()
