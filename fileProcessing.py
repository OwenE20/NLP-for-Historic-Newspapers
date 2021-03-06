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
    from nltk.stem import PorterStemmer
    from spellchecker import SpellChecker
    import nltk
    import pandas as pd
    import pickle

    import random

    def __init__(self, root_dir, target_dir, news_paper, sample_size, isFileSetup, isCorpusBuilt=False):

        symSpell_dictionary = r"C:\Users\Mikes_Surface2\Anaconda3\Lib\site-packages\symspellpy\frequency_dictionary_en_82_765.txt"
        stop_file = r"D:\SeniorProject\ProjectScripts\NLP-for-Historic-Newspapers\stop.pickle"

        self.stemmer = self.PorterStemmer()
        self.spellchecker = self.SymSpell()
        self.spellchecker.load_dictionary(symSpell_dictionary, 0, 1)
        self.word_set = self.spellchecker._words.keys()
        
        self.root = root_dir
        self.target = target_dir
        self.paperName = news_paper
        

        if(isFileSetup == False):
            print("walking")
            self.walkAndProcess()

        try:
            with open(stop_file, 'rb') as file:
                stop = self.pickle.load(file)
                file.close()
        except EOFError:
            stop = set([])
        
        self.stopset = set(self.stopwords.words("english")).union(stop)
        self.spell = self.SpellChecker()

        corpus_filename = r"D:\SeniorProject\ProjectScripts\NLP-for-Historic-Newspapers" + "\\" + news_paper + "\corpus" + ".pickle"
        self.df_filename = r"D:\SeniorProject\ProjectScripts\NLP-for-Historic-Newspapers" + "\\" + news_paper + "\df" + ".pickle"

        if (isCorpusBuilt == False):
            with open(corpus_filename, 'wb') as file:
                print("---- BUILDING CORPUS ----")
                self.corpus = self.buildCorpus(sample_size)
                self.pickle.dump(self.corpus, file)
                file.close()
        else:
            with open(corpus_filename, "rb") as file:
                print("----LOADING CORPUS---")
                self.corpus = self.pickle.load(file)
                file.close()
            with open(self.df_filename, "rb") as file:
                print("----LOADING DF---")
                self.df = self.pickle.load(file)
                file.close()

        # IF PERFORMANCE IS SUBPAR WITH SPELL CHECK, FIND ERA-SPECIFIC CORPUS TO GENERATE WORD FREQUENCIES

    def walkAndProcess(self):
        dates = self.re.compile(r'\d{1,4}')
        # regular expression to find dates within file path, looks for 2-4 digits

        for root, dirs, files in self.os.walk(self.root):
            if (len(files) > 1 and files != None):
                date_list = dates.findall(self.os.path.join(root, files[1]))
                print(date_list)
                newPathName = date_list[0] + date_list[1] + date_list[2] + "_" + date_list[4] +  ".xml"
                newRoot = self.target + "/" + self.paperName + newPathName

                self.shutil.move(self.os.path.join(root, files[1]), newRoot)

    def parse_xml(self, filename):

        date = self.re.compile(r'\d{7,9}')
        index = date.findall(filename)[0]
        blockList = []

        with open(filename) as markupraw:
            opened = self.BeautifulSoup(markupraw, "xml")
        # iterates through xml tree to get content
        for block in opened.find_all('TextBlock'):
            current_block = ''
            for textLine in block:
                for string in textLine:
                    if ('CONTENT' in string.attrs):
                        if (string['CONTENT'] != None):
                            current_block += " " + string['CONTENT']
            blockList.append(current_block)

        return (blockList, index)

    """
    NOT FULL PREPROCESSING, JUST PREPARING TEXT FOR BETTER STORAGE
    RETAINING AS MUCH INFO AS POSSIBLE TO EXPAND PROCESSING OPPORTUNITIES
    
    parameter list1 is an unfiltered list of parsed text
    filtered text is stored in list2
    
    """

    def cleanList(self, list1):
        list2 = []
        for index, st in enumerate(list1):
            print(len(st))
            clean_string = ""
            temp_list = []
            for word in self.nltk.tokenize.word_tokenize(st):
                corrected = "n"
                if (word.isalpha()):
                    if (word in self.spell.unknown([word])):
                        suggestions = self.spellchecker.lookup(word, self.Verbosity.CLOSEST, include_unknown=False)
                        if (len(suggestions) > 0):
                            corrected = suggestions[0].term.lower()
                    elif(len(self.spell.unknown([word])) == 0):
                        corrected = word.lower()
                if(corrected not in self.stopset and len(corrected) > 2):
                    corrected = self.stemmer.stem(corrected)
                    temp_list.append(corrected)
            if (len(temp_list) > 10):
                clean_string = " ".join(temp_list)
                list2.append(clean_string)
    
        return list2

        # sample_size is how many files

    def buildCorpus(self, sample_size):
        fileList = []
        for file in self.os.listdir(self.target):
            fileList.append(self.os.path.join(self.target, file))

        self.df = self.move_to_df(fileList, sample_size)
        with open(self.df_filename, 'wb') as file:
            print("---- BUILDING DF ----")
            self.pickle.dump(self.df, file)
            file.close()
        corpus = []
        for index, data in self.df.iterrows():
            for index, element in enumerate(data[0]):
                corpus.append(element)
        return corpus

    def move_to_df(self, files, sample_size):
        temp_dict = {}
        # takes a random sample of files
        random_files = self.random.sample(files, sample_size)
        count = 1
        for file in random_files:
            print("processing file: " + str(count))
            count = count + 1
            text, date = self.parse_xml(file)
            temp_text = self.cleanList(text)
            temp_dict[str(date)] = [temp_text]
        df = self.pd.DataFrame.from_dict(temp_dict, orient='index')
        print(df.index)
        df.index = self.pd.to_datetime(df.index,errors = "coerce")
        return df

"""
file = r"D:\SeniorProject\FakeCorGazReorganized\FakeCorGaz18990901.xml"
root_FC = r"D:\SeniorProject\FakeCorGaz"
target_FC = r"D:\SeniorProject\FakeCorGazReorganized"
fp_FC = fileProcess(root_FC,target_FC, sample_size=10, news_paper= "FakeCorGaz",isCorpusBuilt = False)
#st = "miaow saycbtdat probably electric lotto tace ball correct pratt pitcher kratoa away nest examined teacher certify cate obtained third grade failed ttsaight army occur prana carnival africa pro live affair star vwatbi anile race qacar chasm albany fraak andrew arnold amine justice coder hoed await action errand fort harding barn mclkiaald mcdonald dvuiy myers goslin allan parker ktley parqae brow john packard pilot stat herbarium uolmao fool dairy ferry heashaw protector bradford boatman pleasure reader troy farsi ceugh syrnp earful season care sudden cold check cough lane trouble beat cleanest sold wlll brown church eloquent impressive lector instructive scholarly popular natal literary tree torn hear admission rsligkus ssaviccs first church service preaching whitaker president willamette university salem sunday junior league devotional epwoith league service abbett usual service held church sunday preaching welcome service vice tomorrow atthepresbvteiian church beat follows preaching pastor subject morning topic modern blindness there reception member connection morning service united prtsbyterian church vice conducted wallace east portland sabbstn junior society teacher prayer welcome service whiteaker clock truly halt live comparatively perfect wing impure condition along scarcely thought utilise forced mention thousand suffering salt rheum serious disorder agony imagined marked success hood sarsanarilla trouble shown advertising column certainly seem justify urging excellent knew disordered everv claim behalf hood sarsanarilla fully backed still proprietor urge merit suffer impure small degree certainly mean include"
#list1 = [st]
#list2 = fp_FC.cleanList(list1)
#print(list2)
"""