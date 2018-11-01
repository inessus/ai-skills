from aip import AipSpeech


API_KEY = "dzli4VACvqd3iXrnIyKahPe6"
SECRET_KEY = "xnB5gpzcPG0QhwioAL4WCSbUmD7Lpgf3"
APP_ID = "14634004"
client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)


# 读取文件
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

# 识别本地文件
client.asr(get_file_content('audio.pcm'), 'pcm', 16000, {'dev_pid': 1737,})