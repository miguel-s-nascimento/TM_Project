from autocorrect import Speller
import nltk
from unidecode import unidecode
from nltk.tokenize.treebank import TreebankWordDetokenizer

def regex_cleaner(raw_text, 
            no_emojis = True, 
            no_hashtags = True,
            hashtag_retain_words = True,
            no_newlines = True,
            no_urls = True,
            no_punctuation = True):
    
    #patterns
    newline_pattern = "(\\n)"
    hashtags_at_pattern = "([#\@@\u0040\uFF20\uFE6B])"
    hashtags_ats_and_word_pattern = "([#@]\w+)"
    emojis_pattern = "([\u2600-\u27FF])"
    url_pattern = "(?:\w+:\/{2})?(?:www)?(?:\.)?([a-z\d]+)(?:\.)([a-z\d\.]{2,})(\/[a-zA-Z\/\d]+)?" ##Note that this URL pattern is *even better*
    punctuation_pattern = "[\u0021-\u0026\u0028-\u0029\u002B\u002C\u002E-\u002F\u003A-\u003F\u005B-\u005F\u2010-\u2028\ufeff`]+"
    apostrophe_pattern = "'(?=[A-Z\s])|(?<=[a-z\.\?\!\,\s])'"
    separated_words_pattern = "(?<=\w\s)([A-Z]\s){2,}"
    ##note that this punctuation_pattern doesn't capture ' this time to allow our tokenizer to separate "don't" into ["do", "n't"]
    
    if no_emojis == True:
        clean_text = re.sub(emojis_pattern,"",raw_text)
    else:
        clean_text = raw_text

    if no_hashtags == True:
        if hashtag_retain_words == True:
            clean_text = re.sub(hashtags_at_pattern,"",clean_text)
        else:
            clean_text = re.sub(hashtags_ats_and_word_pattern,"",clean_text)
        
    if no_newlines == True:
        clean_text = re.sub(newline_pattern," ",clean_text)

    if no_urls == True:
        clean_text = re.sub(url_pattern,"",clean_text)
    
    if no_punctuation == True:
        clean_text = re.sub(punctuation_pattern,"",clean_text)
        clean_text = re.sub(apostrophe_pattern,"",clean_text)

    return clean_text

def lemmatize_all(token, list_pos=["n","v","a","r","s"]):
    
    wordnet_lem = nltk.stem.WordNetLemmatizer()
    for arg_1 in list_pos:
        token = wordnet_lem.lemmatize(token, arg_1)
    return token

def main_pipeline(raw_text, 
                  print_output = True, 
                  no_stopwords = True,
                  custom_stopwords = [],
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

    tokenized_text = [re.sub("'m","am",token) for token in tokenized_text]
    tokenized_text = [re.sub("n't","not",token) for token in tokenized_text]
    tokenized_text = [re.sub("'s","is",token) for token in tokenized_text]

    if no_stopwords == True:
        stopwords = nltk.corpus.stopwords.words("english")
        tokenized_text = [item for item in tokenized_text if item.lower() not in stopwords]
    
    if convert_diacritics == True:
        tokenized_text = [unidecode(token) for token in tokenized_text]

    if lemmatized == True:
        tokenized_text = [lemmatize_all(token, list_pos=list_pos) for token in tokenized_text]
    
    if stemmed == True:
        porterstemmer = nltk.stem.PorterStemmer()
        tokenized_text = [porterstemmer.stem(token) for token in tokenized_text]
 
    if no_stopwords == True:
        tokenized_text = [item for item in tokenized_text if item.lower() not in custom_stopwords]

    if pos_tags_list == "pos_list" or pos_tags_list == "pos_tuples" or pos_tags_list == "pos_dictionary":
        pos_tuples = nltk.tag.pos_tag(tokenized_text)
        pos_tags = [pos[1] for pos in pos_tuples]
    
    if lowercase == True:
        tokenized_text = [item.lower() for item in tokenized_text]

    if print_output == True:
        print(raw_text)
        print(tokenized_text)
    
    if pos_tags_list == "pos_list":
        return (tokenized_text, pos_tags)
    elif pos_tags_list == "pos_tuples":
        return pos_tuples   
    if word_correction == True:
        print(tokenized_text)
        spell = Speller()
        tokenized_text= [spell(item) for item in tokenized_text]


    else:
        if tokenized_output == True:
            return tokenized_text
        else:
            detokenizer = TreebankWordDetokenizer()
            detokens = detokenizer.detokenize(tokenized_text)
            return str(detokens)



####################################
#giberish detection
import re
import math


def split_in_chunks(text, chunk_size):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    if len(chunks) > 1 and len(chunks[-1]) < 10:
        chunks[-2] += chunks[-1]
        chunks.pop(-1)
    return chunks


def unique_chars_per_chunk_percentage(text, chunk_size):
    chunks = split_in_chunks(text, chunk_size)
    unique_chars_percentages = []
    for chunk in chunks:
        total = len(chunk)
        unique = len(set(chunk))
        unique_chars_percentages.append(unique / total)
    return sum(unique_chars_percentages) / len(unique_chars_percentages) * 100


def vowels_percentage(text):
    vowels = 0
    total = 0
    for c in text:
        if not c.isalpha():
            continue
        total += 1
        if c in "aeiouAEIOU":
            vowels += 1
    if total != 0:
        return vowels / total * 100
    else:
        return 0


def word_to_char_ratio(text):
    chars = len(text)
    words = len([x for x in re.split(r"[\W_]", text) if x.strip() != ""])
    return words / chars * 100


def deviation_score(percentage, lower_bound, upper_bound):
    if percentage < lower_bound:
        return math.log(lower_bound - percentage, lower_bound) * 100
    elif percentage > upper_bound:
        return math.log(percentage - upper_bound, 100 - upper_bound) * 100
    else:
        return 0


def classify(text):
    if text is None or len(text) == 0:
        return 0.0
    ucpcp = unique_chars_per_chunk_percentage(text, 35)
    vp = vowels_percentage(text)
    wtcr = word_to_char_ratio(text)

    ucpcp_dev = max(deviation_score(ucpcp, 45, 50), 1)
    vp_dev = max(deviation_score(vp, 35, 45), 1)
    wtcr_dev = max(deviation_score(wtcr, 15, 20), 1)

    return max((math.log10(ucpcp_dev) + math.log10(vp_dev) +
                math.log10(wtcr_dev)) / 6 * 100, 1)