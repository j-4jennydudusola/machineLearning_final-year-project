from urllib import request
import re

def retrieve_text_input(url):
    html = request.urlopen(url)
    # extract html from the response object returned
    raw_input = html.read().decode('utf8')
    # import beautiful soup package to use methods to extract data form html tags
    from bs4 import BeautifulSoup
    raw_text = BeautifulSoup(raw_input, 'html.parser').get_text()

    return raw_text

def compute_regex(text):

    phoneNumRegex = re.compile(r'\+?\d{2,4}[-.\s]?\d{2,6}[-.\s]?\d{3,6}')
    search = phoneNumRegex.findall(text)
    result = [number for number in search if re.search(phoneNumRegex, number)]

    if search is not None:
        print("Found a match!")
        for number in search:
            print("Telephone: ", number)
    else:
        print("No numbers found!")



def main():
    url = input("Enter url here: ")
    text = retrieve_text_input(url)
    compute_regex(text)



if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)