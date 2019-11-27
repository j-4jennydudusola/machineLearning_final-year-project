from nltk import re, unique_list, pos_tag
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from tabulate import tabulate


def retrieve_text(url):
    import requests
    # make the get request from website
    request = requests.get(url)
    # extract html from the response object returned
    html = request.text

    # import beautiful soup package to use methods to extract data form html tags
    from bs4 import BeautifulSoup
    extractor = BeautifulSoup(html, features="html.parser")
    # retrieve text from the html page
    text = extractor.find('article').get_text()

    return text


def tokenize_and_normalise(text):

    # find text tokens using tokenizer
    text_tokens = word_tokenize(text)
    # define words as alphanumeric
    text_nopunct = [word for word in text_tokens if re.search("\w", word)]

    return text_nopunct

def lemmitization(text):
    wordnet_lemmatizer = WordNetLemmatizer()
    lem_tokens = [wordnet_lemmatizer.lemmatize(word.lower()) for word in text]


    return lem_tokens



def main():
    url = 'https://www.theguardian.com/music/2018/oct/19/while-my-guitar-gently-weeps-beatles-george-harrison'
    text = retrieve_text(url)
    # find the types using unique list method to find unique words in text
    text_types = unique_list(tokenize_and_normalise(text))

    print("\---------Question 1-----------")
    print()
    print("This text contains tokens before lemmatization: " + str(len(tokenize_and_normalise(text))))
    print("This text contains types before lemmatization: " + str(len(text_types)))
    print()

    lem_types = unique_list(lemmitization(tokenize_and_normalise(text)))
    print("This text contains tokens after lemmatization: " + str(len(lemmitization(tokenize_and_normalise(text)))))
    print("This text contains types after lemmatization: " + str(len(lem_types)))
    print("\n---------Question 2-----------")
    print(tabulate(pos_tag(tokenize_and_normalise(text)), headers=['Word', 'POS Tag']))
    print("\n")
    print(tabulate(pos_tag(lemmitization(tokenize_and_normalise(text))), headers=['Lemmitized Word', 'POS Tag']))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
