import re

def read_txt_file(text_file):

    try:
        with open(text_file,'r',encoding = 'utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"{text_file} File Not Found")
        return 

    # remove punctuation 
    cleaned = re.sub(r'[^\w\s]','',text) # this will only allow \w-> words, \s-> spaces in text other will be removed with "" which is nothing

    # removes extra spaces
    cleaned = cleaned.strip()

    # save clear text 
    with open('cleaned_text.txt','w',encoding = 'utf-8') as f:
        f.write(cleaned)

    return cleaned
    
read_txt_file("text.txt")
    