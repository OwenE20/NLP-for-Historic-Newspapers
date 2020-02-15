# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 15:45:06 2019

@author: Mikes_Surface2
"""

class fileProcess:
    import os
    import re 
    import shutil
    from symspellpy import SymSpell, Verbosity
    from bs4 import BeautifulSoup
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    import nltk
    import pandas as pd

    import random

    def __init__(self,root_dir,target_dir,news_paper):

        symSpell_dictionary = r"C:\Users\Mikes_Surface2\Anaconda3\Lib\site-packages\symspellpy\frequency_dictionary_en_82_765.txt"

        self.spellchecker = self.SymSpell()
        self.spellchecker.load_dictionary(symSpell_dictionary,0,1)
        self.word_set = self.spellchecker._words.keys()

        self.root = root_dir
        self.target = target_dir
        self.paperName = news_paper
        self.stopset = set(self.stopwords.words("english"))
        self.lemmatizer = self.WordNetLemmatizer()

        
        #IF PERFORMANCE IS SUBPAR WITH SPELL CHECK, FIND ERA-SPECIFIC CORPUS TO GENERATE WORD FREQUENCIES 
      
    
    def walkAndProcess(self):
        dates = self.re.compile(r'\d{2,4}')
        #regular expression to find dates within file path, looks for 2-4 digits
        
        
        for root,dirs,files in self.os.walk(self.root):
            if(len(files) > 1 and files != None):
                date_list = dates.findall(self.os.path.join(root,files[1]))
                print(date_list)
                newPathName = date_list[0] + date_list[1] + date_list[2] + ".xml"
                newRoot = self.target + "/" + self.paperName + newPathName
                self.shutil.move(self.os.path.join(root,files[1]), newRoot)
              
            
    def parse_xml(self,filename):

        date = self.re.compile(r'\d{7,9}')
        index = date.findall(filename)[0]
        blockList = []
    
        with open(filename) as markupraw:
            opened = self.BeautifulSoup(markupraw, "xml")
        #iterates through xml tree to get content 
        for block in opened.find_all('TextBlock'):
            current_block = ''
            for textLine in block:
                for string in textLine:
                    if('CONTENT' in string.attrs): 
                        if(string['CONTENT'] != None):
                            current_block += " " + string['CONTENT']
            blockList.append(current_block)
    
        return (blockList,index)
    


    """
    NOT FULL PREPROCESSING, JUST PREPARING TEXT FOR BETTER STORAGE
    RETAINING AS MUCH INFO AS POSSIBLE TO EXPAND PROCESSING OPPORTUNITIES
    
    parameter list1 is an unfiltered list of parsed text
    filtered text is stored in list2
    
    """
    def cleanList(self,list1):
        list2 = []
        for index, st in enumerate(list1):
            clean_string = ""
            temp_list = []
            for word in self.nltk.tokenize.word_tokenize(st):
                corrected = ""
                if(word.isalpha()):
                   suggestions = self.spellchecker.lookup(word, self.Verbosity.CLOSEST, max_edit_distance=2,include_unknown=True)

                   if(len(suggestions) > 1):
                       corrected = suggestions[0].term

                   else:
                       print(word)
                       if(word in self.word_set or self.nltk.pos_tag([word])[0][1] == "NN" or self.nltk.pos_tag([word])[0][1] == "NNP"):
                            print(word)
                            corrected = word
                       else:
                            corrected = "n"

                   if(corrected not in self.stopset and len(corrected) > 3):
                       corrected = self.lemmatizer.lemmatize(corrected.lower())
                       temp_list.append(corrected)
            if(len(temp_list) > 25):
                clean_string = " ".join(temp_list)
                print("cleaned a string")
                list2.append(clean_string)
            
        return list2
            
    

    def move_to_df(self,files,sample_size):
        temp_dict = {}
        #takes a random sample of files
        random_files = self.random.sample(files,sample_size)
        for file in random_files:
            text, date = self.parse_xml(file)
            #random sample within the documents: 3/5ths of the documents currently
            size = (3 * len(text)//5)
            temp_text = self.cleanList(self.random.sample(text,size))
            temp_dict[str(date)] = [temp_text]
        df = self.pd.DataFrame.from_dict(temp_dict,orient = 'index')
        df.index = self.pd.to_datetime(df.index)
        return df
        
        

file = "D:\SeniorProject\FakeCorGazReorganized\FakeCorGaz18990901.xml"
root_FC = r"D:\SeniorProject\FakeCorGaz"
target_FC = r"D:\SeniorProject\FakeCorGazReorganized"
fp_FC = fileProcess(root_FC,target_FC, "FakeCorGaz")
list1 = fp_FC.parse_xml(file)[0]
clean = fp_FC.cleanList(list1)
print(clean)

