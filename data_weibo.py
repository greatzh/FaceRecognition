# -*- coding: utf-8 -*-
import urllib.request
import json
import requests
import os

path = './raw_data/'

#id = liuchang!!!'1299491507'
#id = zhengshuang 5574201257

#id = '5574201257'
print('请输入微博ID，为一串数字')
id = input()
proxy_addr = "122.241.72.191:808"
pic_num = 0
# print('请输入保存图片文件的文件夹名字，请使用英文字符，例如“刘畅”可以输入“liuchang”')
# weibo_name = input()
# print('请输入是否要自定义保存图片路径，默认为在F盘下新建一个您刚刚输入的文件夹名字进行保存，如果您希望自定义保存文件夹，请输入数字2，如果选择使用默认文件夹，请输入数字1,建议使用默认路径')
# choose = int(input())
# if choose == 1:
#     print('您选择使用默认文件夹查看图片请前往F：\'', weibo_name)
# else:
#     print('您选择使用自定义保存路径，请确保该路径已经存在，接下来请您按照以下格式输入保存路径，例如E:\python3\StarFaceRecognition\ori_data,您需要输入的则是E:\\python3\\StarFaceRecognition\\ori_data')
#     path = input()
weibo_name = "liuchang_weibo"

1299491507
def use_proxy(url, proxy_addr):
    req = urllib.request.Request(url)
    req.add_header("User-Agent",
                   "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.221 Safari/537.36 SE 2.X MetaSr 1.0")
    proxy = urllib.request.ProxyHandler({'http': proxy_addr})
    opener = urllib.request.build_opener(proxy, urllib.request.HTTPHandler)
    urllib.request.install_opener(opener)
    data = urllib.request.urlopen(req).read().decode('utf-8', 'ignore')
    return data


def get_containerid(url):
    data = use_proxy(url, proxy_addr)
    content = json.loads(data).get('data')
    for data in content.get('tabsInfo').get('tabs'):
        if (data.get('tab_type') == 'weibo'):
            containerid = data.get('containerid')
    return containerid


def get_userInfo(id):
    url = 'https://m.weibo.cn/api/container/getIndex?type=uid&value=' + id
    data = use_proxy(url, proxy_addr)
    content = json.loads(data).get('data')
    profile_image_url = content.get('userInfo').get('profile_image_url')
    description = content.get('userInfo').get('description')
    profile_url = content.get('userInfo').get('profile_url')
    verified = content.get('userInfo').get('verified')
    guanzhu = content.get('userInfo').get('follow_count')
    name = content.get('userInfo').get('screen_name')
    fensi = content.get('userInfo').get('followers_count')
    gender = content.get('userInfo').get('gender')
    urank = content.get('userInfo').get('urank')
    print("微博昵称：" + name + "\n" + "微博主页地址：" + profile_url + "\n" + "微博头像地址：" + profile_image_url + "\n" + "是否认证：" + str(
        verified) + "\n" + "微博说明：" + description + "\n" + "关注人数：" + str(guanzhu) + "\n" + "粉丝数：" + str(
        fensi) + "\n" + "性别：" + gender + "\n" + "微博等级：" + str(urank) + "\n")


def get_weibo(id, file):
    global pic_num
    i = 1
    while True:
        url = 'https://m.weibo.cn/api/container/getIndex?type=uid&value=' + id
        weibo_url = 'https://m.weibo.cn/api/container/getIndex?type=uid&value=' + id + '&containerid=' + get_containerid(
            url) + '&page=' + str(i)
        try:
            data = use_proxy(weibo_url, proxy_addr)
            content = json.loads(data).get('data')
            cards = content.get('cards')
            if (len(cards) > 0):
                for j in range(len(cards)):
                    print("-----正在爬取第" + str(i) + "页，第" + str(j) + "条微博------")
                    card_type = cards[j].get('card_type')
                    if (card_type == 9):
                        mblog = cards[j].get('mblog')
                        attitudes_count = mblog.get('attitudes_count')
                        comments_count = mblog.get('comments_count')
                        created_at = mblog.get('created_at')
                        reposts_count = mblog.get('reposts_count')
                        scheme = cards[j].get('scheme')
                        text = mblog.get('text')
                        if mblog.get('pics') != None:
                            # print(mblog.get('original_pic'))
                            # print(mblog.get('pics'))
                            pic_archive = mblog.get('pics')
                            for _ in range(len(pic_archive)):
                                pic_num += 1
                                print(pic_archive[_]['large']['url'])
                                imgurl = pic_archive[_]['large']['url']
                                img = requests.get(imgurl)
                                f = open(path + weibo_name + '\\' + str(pic_num) + str(imgurl[-4:]),
                                         'ab')  # 存储图片，多媒体文件需要参数b（二进制文件）
                                f.write(img.content)  # 多媒体存储content
                                f.close()

                        with open(file, 'a', encoding='utf-8') as fh:
                            fh.write("----第" + str(i) + "页，第" + str(j) + "条微博----" + "\n")
                            fh.write("微博地址：" + str(scheme) + "\n" + "发布时间：" + str(
                                created_at) + "\n" + "微博内容：" + text + "\n" + "点赞数：" + str(
                                attitudes_count) + "\n" + "评论数：" + str(comments_count) + "\n" + "转发数：" + str(
                                reposts_count) + "\n")
                i += 1
            else:
                break
        except Exception as e:
            print(e)
            pass


if __name__ == "__main__":
    if os.path.isdir(path + weibo_name):
        pass
    else:
        os.mkdir(path + weibo_name)
    file = path + weibo_name + '\\' + weibo_name + ".txt"
    get_userInfo(id)
    get_weibo(id, file)
