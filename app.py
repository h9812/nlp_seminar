MODEL_FILE_PATH = "topic_detection_fasttext_v2.bin"

import nltk
nltk.download('punkt')
from nltk.tokenize import MWETokenizer, word_tokenize, RegexpTokenizer
import re
import nltk
import unicodedata


class PostPreprocess:
    def __init__(self):
        
        self.multiple_punctuation_pattern = re.compile(r"([\"\.\?\!\,\:\;\-])(?:[\"\.\?\!\,\:\;\-]){1,}")
        self.word_tokenizer = MWETokenizer(separator='')
        self.multiple_emoji_pattern = re.compile(u"(["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                u"\u00a9"
                u"\u00ae"
                u"\u2000-\u3300"
                "]){1,}", flags= re.UNICODE )

        self.normalizer = {'òa': 'oà',
                    'óa': 'oá',
                    'ỏa': 'oả',
                    'õa': 'oã',
                    'ọa': 'oạ',
                    'òe': 'oè',
                    'óe': 'oé',
                    'ỏe': 'oẻ',
                    'õe': 'oẽ',
                    'ọe': 'oẹ',
                    'ùy': 'uỳ',
                    'úy': 'uý',
                    'ủy': 'uỷ',
                    'ũy': 'uỹ',
                    'ụy': 'uỵ',
                    'Ủy': 'Uỷ'}
        self.correct_mapping = {
            "m": "mình",
            "mik": "mình",
            "ko": "không",
            "k": " không ",
            "kh": "không",
            "khong": "không",
            "kg": "không",
            "khg": "không",
            "tl": "trả lời",
            "r": "rồi",
            "ok": "tốt",
            "dc": "được",
            "vs": "với",
            "đt": "điện thoại",
            "thjk": "thích",
            "thik": "thích",
            "qá": "quá",
            "trể": "trễ",
            "bgjo": "bao giờ",
            "''": '"',
            "``": '"'
        }

   

    def normalize_text(self, text):
        for absurd, normal in self.normalizer.items():
            text = text.replace(absurd, normal)

        # for l in vn_location:
        #   text = text.replace(l, ' location ')

        return text

    def tokmap(self, tok):
        if tok.lower() in self.correct_mapping:
            return self.correct_mapping[tok.lower()]
        else:
            return tok

    def preprocess(self, text):
        text = self.multiple_emoji_pattern.sub(r"\g<1> ", text) # \g<1>
        text = self.multiple_punctuation_pattern.sub(r" \g<1> ", text)
        text = unicodedata.normalize("NFC", text)
        text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b(\/)?', 'url', text)
        text = re.sub("\.", " . ", text)
        text = re.sub("'", "' ", text)
        text = re.sub('"', '" ', text)
        text = re.sub('/', ' / ', text)
        text = re.sub('-', ' - ', text)
        text = re.sub(',', ' , ', text)
        text = re.sub(r'\s{2,}', ' ', text)
        text = self.normalize_text(text)
        # text = re.sub(r'\#[^\s]+', ' hastag ', text)
        text = re.sub(r'(|\s)([\d]+k)(\s|$)', ' cureency_k ', text)
        text = re.sub(r'(([\d]{2,4}\s){2,}([\d]+)?|(09|01|[2|6|8|9]|03)+([0-9]{8})\b)', ' phone_number ', text)
        text = re.sub(r'\d', "_digit", text)
        tokens = self.word_tokenizer.tokenize(word_tokenize(text))
        tokens = list(map(self.tokmap, tokens))
        # return tokens
        return ' '.join(tokens)

import fasttext

class PostClassifier:
    def __init__(self, model_file=MODEL_FILE_PATH):
        self.model = fasttext.load_model(model_file)
        self.preprocessor = PostPreprocess()
    
    def predict(self, text):
        text = self.preprocessor.preprocess(text)
        result = self.model.predict(text)
        topic = result[0][0]
        score = round(result[1][0], 2)
        return topic, score

classifier = PostClassifier()

from flask import Flask, request, render_template
import traceback

app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        try:
            post = request.form['post']
            if not post or len(post) <= 50:
                return render_template('index.html', error='Hãy nhập nhiều hơn 50 ký tự',  post=post)
            topic, score = classifier.predict(post)
            return render_template('index.html', topic=topic, score=score, post=post)
        except:
            traceback.print_exc()
            return render_template('index.html', error='Chẳng biết lỗi gì, thử lại xem sao',  post=post)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port='8000')
    

