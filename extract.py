import sys
import os
from xml.dom import minidom
from bs4 import BeautifulSoup
import sys
import json
import operator

root = sys.argv[1]

outfile = open("out/raw.json", "w")
data = []
count = {}
for subdir, dirs, files in os.walk(root):
    for _file in files:
        if _file == ".content.xml":
            # print (os.path.join(subdir, _file))
            try:
                dom = minidom.parse(os.path.join(subdir, _file))
                text = dom.getElementsByTagName("jcr:content")
                mText = text[0].attributes['marqueeText'].value
                comp = dom.getElementsByTagName("textcomp")
                textcomp = comp[0].attributes["text"].value

                soup = BeautifulSoup(mText, 'html.parser')
                mText = soup.get_text()
                soup = BeautifulSoup(textcomp, 'html.parser')
                textcomp = soup.get_text()

                textcomp = str(''.join([i if ord(i) < 128 else '' for i in textcomp.replace("\n", ". ")])).replace("..", ".")
                mText = str(''.join([i if ord(i) < 128 else '' for i in mText.replace("\n", ". ")])).replace("..", ".")

                textcomp = ". ".join(filter(None, textcomp.split(". ")))
                mText = ". ".join(filter(None, mText.split(". ")))

                for i in textcomp.split():
                    if i in count:
                        count[i] += 1
                    else:
                        count[i] = 1
                for i in mText.split():
                    if i in count:
                        count[i] += 1
                    else:
                        count[i] = 1
                data.append([textcomp, mText])
            except:
                pass
json.dump(data, outfile)

vocab = sorted(count.items(), key=operator.itemgetter(1), reverse=True)
vocabfile = open("out/vocab", "w")
for entry in vocab:
    vocabfile.write(entry[0] + " " + str(entry[1]) + "\n")
