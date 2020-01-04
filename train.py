import nltk
nltk.download('punkt')
from nltk.tokenize import MWETokenizer, word_tokenize, RegexpTokenizer
import re
import nltk
import unicodedata

multiple_punctuation_pattern = re.compile(r"([\"\.\?\!\,\:\;\-])(?:[\"\.\?\!\,\:\;\-]){1,}")
word_tokenizer = MWETokenizer(separator='')
multiple_emoji_pattern = re.compile(u"(["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\u00a9"
        u"\u00ae"
        u"\u2000-\u3300"
        "]){1,}", flags= re.UNICODE )

normalizer = {'òa': 'oà',
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
correct_mapping = {
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

def normalize_text(text):
  for absurd, normal in normalizer.items():
    text = text.replace(absurd, normal)

  # for l in vn_location:
  #   text = text.replace(l, ' location ')

  return text

def tokmap(tok):
  if tok.lower() in correct_mapping:
      return correct_mapping[tok.lower()]
  else:
      return tok

def preprocess(text):
  global i
  text = multiple_emoji_pattern.sub(r"\g<1> ", text) # \g<1>
  text = multiple_punctuation_pattern.sub(r" \g<1> ", text)
  text = unicodedata.normalize("NFC", text)
  text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b(\/)?', 'url', text)
  text = re.sub("\.", " . ", text)
  text = re.sub("'", "' ", text)
  text = re.sub('"', '" ', text)
  text = re.sub('/', ' / ', text)
  text = re.sub('-', ' - ', text)
  text = re.sub(',', ' , ', text)
  text = re.sub(r'\s{2,}', ' ', text)
  text = normalize_text(text)
  # text = re.sub(r'\#[^\s]+', ' hastag ', text)
  text = re.sub(r'(|\s)([\d]+k)(\s|$)', ' cureency_k ', text)
  text = re.sub(r'(([\d]{2,4}\s){2,}([\d]+)?|(09|01|[2|6|8|9]|03)+([0-9]{8})\b)', ' phone_number ', text)
  text = re.sub(r'\d', "_digit", text)
  tokens = word_tokenizer.tokenize(word_tokenize(text))
  tokens = list(map(tokmap, tokens))
  # return tokens
  return ' '.join(tokens)

with open('topic_detection_test.v1.0.txt') as f:
    topic_detection_test = f.read().strip().split('\n')
    topic_detection_test = [ preprocess(e) for e in topic_detection_test]

with open('topic_detection_fasttext_test.v1.0.txt', 'w') as f:
  f.write('\n'.join(topic_detection_test))

with open('topic_detection_train.v1.0.txt') as f:
  topic_detection_train = f.read().strip().split('\n')
  print(topic_detection_train[:2])
  topic_detection_train = [ line.split(' ',1) for line in topic_detection_train]
  print(topic_detection_train[:2])
  topic_detection_train = [ [lables, preprocess(descriptions)] for lables, descriptions in topic_detection_train]
  print(topic_detection_train[:2])
  topic_detection_train = [ ' '.join(e) for e in topic_detection_train]
print(topic_detection_train[:2])
with open('topic_detection_fasttext_train.v1.0.txt', 'w') as f:
  f.write('\n'.join(topic_detection_train))

import fasttext

#Train the model
model = fasttext.train_supervised(input="topic_detection_fasttext_train.v1.0.txt", lr=0.22, wordNgrams=3) #  epoch=5)# dim=10)
print(type(model))
model.save_model("topic_detection_fasttext.bin")
prediction = []

for i in range(len(topic_detection_test)):
  prediction.append(model.predict(topic_detection_test[i])[0][0] )
print(len(prediction))
