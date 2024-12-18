import re
from autocorrect import Speller
import nltk
from unidecode import unidecode
from nltk.tokenize.treebank import TreebankWordDetokenizer
from utils.gibberish_detector import classify_gibberish
import contractions


def regex_cleaner(raw_text, 
            no_emojis = True, 
            no_hashtags = True,
            hashtag_retain_words = True,
            no_newlines_carr_return = True,
            no_urls = True,
            no_punctuation = True,
            no_contractions = True,
            no_numbers = True):
    
    #patterns
    newline_carr_return_pattern = "[\r\n]+"
    hashtags_at_pattern = "([#\@@\u0040\uFF20\uFE6B])"
    hashtags_ats_and_word_pattern = "([#@]\w+)"
    emojis_pattern = "(\ud83c[\ud000-\udfff]|\ud83d[\ud000-\udfff]|\ud83e[\ud000-\udfff])"
    url_pattern = "(?:https?://)?(?:www\.)?[a-zA-Z0-9.-]+\.(?:com)(?:/[a-zA-Z0-9./?=&_-]*)?"
    punctuation_pattern = "[\u0021-\u0026\u0028-\u0029\u002B\u002C\u002E-\u002F\u003A-\u003F\u005B-\u005F\u2010-\u2028\ufeff`]+"
    apostrophe_pattern = "'(?=[A-Z\s])|(?<=[a-z\.\?\!\,\s])'"
    numbers_pattern = "\d+"
    
    
    if no_emojis:
        clean_text = re.sub(emojis_pattern,"",raw_text)
    else:
        clean_text = raw_text

    if no_hashtags:
        if hashtag_retain_words:
            clean_text = re.sub(hashtags_at_pattern,"",clean_text)
        else:
            clean_text = re.sub(hashtags_ats_and_word_pattern,"",clean_text)
        
    if no_newlines_carr_return:
        clean_text = re.sub(newline_carr_return_pattern," ",clean_text)

    if no_urls:
        clean_text = re.sub(url_pattern,"",clean_text)

    if no_contractions:
        clean_text = contractions.fix(clean_text) # before apostrophe removal

    if no_punctuation:
        clean_text = re.sub(punctuation_pattern," ",clean_text)
        clean_text = re.sub(apostrophe_pattern," ",clean_text)
        clean_text = re.sub("\s{2,}", " ", clean_text) # Eliminate multiple spaces
        
    if no_numbers:
        clean_text = re.sub(numbers_pattern, "", clean_text)
    
    return clean_text

def lemmatize_all(token, list_pos=["n","v","a","r","s"]):
    wordnet_lem = nltk.stem.WordNetLemmatizer()
    for arg_1 in list_pos:
        token = wordnet_lem.lemmatize(token, arg_1)
    return token

def main_pipeline(raw_text,  
                  no_stopwords = True,
                  custom_stopwords = [],
                  exception_stopwords = [],
                  convert_diacritics = True, 
                  lowercase = True, 
                  lemmatized = True,
                  list_pos = ["n","v","a","r","s"],
                  stemmed = False, 
                  pos_tags_list = "no_pos",
                  tokenized_output = False,
                  word_correction=False,
                  **kwargs):
    
    """Preprocess strings according to the parameters"""

    clean_text = regex_cleaner(raw_text, **kwargs)

    tokenized_text = nltk.tokenize.word_tokenize(clean_text)
    
    if no_stopwords:
        stopwords = nltk.corpus.stopwords.words("english")
        stopwords = list(set(stopwords) - set(exception_stopwords))
        tokenized_text = [item for item in tokenized_text if item.lower() not in stopwords]
    
    if convert_diacritics:
        tokenized_text = [unidecode(token) for token in tokenized_text]

    if lemmatized:
        tokenized_text = [lemmatize_all(token, list_pos=list_pos) for token in tokenized_text]
    
    if stemmed:
        porterstemmer = nltk.stem.PorterStemmer()
        tokenized_text = [porterstemmer.stem(token) for token in tokenized_text]
 
    if no_stopwords:
        tokenized_text = [item for item in tokenized_text if item.lower() not in custom_stopwords]

    if pos_tags_list == "pos_list" or pos_tags_list == "pos_tuples" or pos_tags_list == "pos_dictionary":
        pos_tuples = nltk.tag.pos_tag(tokenized_text)
        pos_tags = [pos[1] for pos in pos_tuples]
    
    if lowercase:
        tokenized_text = [item.lower() for item in tokenized_text]
    
    if pos_tags_list == "pos_list":
        return (tokenized_text, pos_tags)
    elif pos_tags_list == "pos_tuples":
        return pos_tuples   
    if word_correction:
        spell = Speller()
        tokenized_text= [spell(item) for item in tokenized_text]

    else:
        if tokenized_output == True:
            return tokenized_text
        else:
            detokenizer = TreebankWordDetokenizer()
            detokens = detokenizer.detokenize(tokenized_text)
            return str(detokens)