# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 15:45:06 2019

@author: Mikes_Surface2
"""

class fileProcess:
    import os
    import re 
    import shutil
    import string
    from bs4 import BeautifulSoup
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    import nltk
    import pandas as pd
    from spellchecker import SpellChecker
    

    def __init__(self,root_dir,target_dir,news_paper):
        self.root = root_dir
        self.target = target_dir
        self.paperName = news_paper
        self.stopset = set(self.stopwords.words("english"))
        self.lemmatizer = self.WordNetLemmatizer()
        self.spell = self.SpellChecker()
        #IF PERFORMANCE IS SUBPAR WITH SPELL CHECK, FIND ERA-SPECIFIC CORPUS TO GENERATE WORD FREQUENCIES 
      
    
    def walkAndProcess(self):
        dates = self.re.compile(r'\d{2,4}')
        #regular expression to find dates within file path, looks for 2-4 digits
        
        
        for root,dirs,files in self.os.walk(self.root):
            if(len(files) > 1 and files != None):
                date_list = dates.findall(self.os.path.join(root,files[1]))
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
    def cleanList(self,list1,list2):
        for index, st in enumerate(list1):
            clean_string = ""
            temp_list = []
            for word in self.nltk.tokenize.word_tokenize(st):
                if(word.isalpha()):
                    corrected_word = word
                    if(word in self.spell.unknown(word)):
                        corrected_word = self.spell.correction(word)
                        if(corrected_word in self.spell.unknown(corrected_word)):
                            #This runs if the word is essentially nonsense: best to get rid of it
                            corrected_word = "n"
                    if(corrected_word not in self.stopset and len(word) > 3):
                        good_word = self.lemmatizer.lemmatize(corrected_word.lower())
                        temp_list.append(good_word)
                    
            if(len(temp_list) > 25):
                for word in temp_list:
                    clean_string += " " + word
                list2.append(clean_string)
            
            

    def move_to_df(self,files):
        temp_dict = {}
        for file in files:
            temp_text = []
            text, date = self.parse_xml(file)
            self.cleanList(text, temp_text)
            temp_dict[str(date)] = [temp_text]
        df = self.pd.DataFrame.from_dict(temp_dict,orient = 'index')
        df.index = self.pd.to_datetime(df.index)
        return df
        
        
    def getTarget(self):
        return self.target



