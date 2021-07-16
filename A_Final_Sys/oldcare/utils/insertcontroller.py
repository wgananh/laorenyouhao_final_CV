import time
from ..utils import getfromURL


class InsertController:
    def __init__(self):
        # self.emotion_label = ['anger', 'disgust', 'fear', 'happy', 'normal', 'sad', 'surprised']
        self.emotion = [0, 0, 0, 0, 0, 0, 0, 1]

    def emotion_add(self, index):
        self.emotion[index] += 1

    # def HttpRequest(self, apiname, controlname, operate, jtype, inputdata):
    #     headers = {"content-Type": "application/json"}
    #     datas = json.JSONEncoder().encode(inputdata)
    #     url = "http://192.168.43.214:88/api/" + apiname + "/" + controlname + "/" + operate
    #     if jtype == "get":
    #         r = requests.get(url, data=datas, headers=headers)
    #     if jtype == "post":
    #         r = requests.post(url, data=datas, headers=headers)
    #
    #     jsonobj = json.loads(r.text)
    #     return jsonobj

    def insert(self, old_name):
        if self.emotion[3] >= 5:
            most_emotion_index = 3
        else:
            most_emotion_index = self.emotion.index(max(self.emotion))
        inputdata = {}
        inputdata["laorenName"] = old_name
        inputdata["emotionType"] = most_emotion_index + 1
        inputdata["date"] = time.strftime("%Y-%m-%d %H:%M:%S")
        result = getfromURL.HttpRequest("laoren", "health", "emotion", "post", inputdata)
        print(result['msg'])
        # result = self.HttpRequest("laoren", "health", "emotion", "post", inputdata)
