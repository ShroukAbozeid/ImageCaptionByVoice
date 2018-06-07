from gtts import gTTS
from playsound import playsound


def convert_to_voice(caption):
   language = 'en'
   myobj = gTTS(text=caption, lang=language, slow=False)
   myobj.save("welcome.mp3")
   playsound('welcome.mp3')