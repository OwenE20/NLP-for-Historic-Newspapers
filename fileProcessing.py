# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 15:45:06 2019

@author: Mikes_Surface2
"""

class fileProcess:
    import os
    import re 
    import shutil
    from bs4 import BeautifulSoup
    from nltk.corpus import stopwords
    from nltk.tokenize import PunktSentenceTokenizer
    from nltk.stem import PorterStemmer
    from nltk.corpus import state_union
    

    def __init__(self,root_dir,target_dir,news_paper):
        self.root = root_dir
        self.target = target_dir
        self.paperName = news_paper
        df_strings = pd.DataFrame(index = ["Dates"], columns = ["TextBlockList"])
        train_text = self.state_union.raw("2006-GWBush.txt")
        self.custom_tokenizer = self.PunktSentenceTokenizer(train_text)
        
    
    def walkAndProcess(self):
        dates = self.re.compile(r'\d{2,4}')
        #regular expression to find dates within file path, looks for 2-4 digits
        
        
        for root,dirs,files in self.os.walk(self.root):
            if(len(files) > 1 and files != None):
                date_list = dates.findall(self.os.path.join(root,files[1]))
                newPathName = date_list[0] + date_list[1] + date_list[2] + ".xml"
                newRoot = target_dir + "/" + self.paperName + newPathName
                self.shutil.move(self.os.path.join(root,files[1]), newRoot)
              
            
            
    """
    TODO: RETURN EMBEDDED DATE TOO, PARSE TOUPLE IN PROCESS METHOD
    """
    def parse_xml(self,filename):
    
        #date = self.re.compile(r'\d{7,9}')
        #index = date.findall(filename)
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
        return blockList
    


    """
    NOT FULL PREPROCESSING, JUST PREPARING TEXT FOR BETTER STORAGE
    RETAINING AS MUCH INFO AS POSSIBLE TO EXPAND PROCESSING OPPORTUNITIES
    """
    def cleanList(self,list1):
        for i,x in enumerate(list1):
            if(len(str(x)) == None or len(str(x)) == 0):
                list1.remove(x)
        stop_words = set(self.stopwords.words("english"))
        
        temp_sent = ""
        for i,x in enumerate(list1):
            words = self.custom_tokenizer.tokenize(x)
            words = str([w for w in words if not w in stop_words]).strip("[]")
            words = words.lower()
            
                    
            
            
            
            

 



import pandas as pd



print(list1)



root_dir = r"D:\SeniorProject\testDir"
target_dir = r"D:\SeniorProject\CorGazReorganized"
news_name = "CorGaz" 
name = "D:\SeniorProject\CorGazReorganized/CorGaz18991027.xml"

fp = fileProcess(root_dir,target_dir, "CorGaz")


list1 = fp.parse_xml(name)
fp.cleanList(list1)


