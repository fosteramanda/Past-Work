""" This module is python practice exercises to cover more advanced topics.
	Put the code for your solutions to each exercise in the appropriate function.
    Remove the 'pass' keyword when you implement the function.
	DON'T change the names of the functions!
	You may change the names of the input parameters.
	Put your code that tests your functions in the if __name__ == "__main__": section
	Don't forget to regularly commit and push to github.
    Please include an __author__ comment so I can tell whose code this is.
"""
__author__ = ""
__version__ = 4.1

import random

# List Comprehension Practice

def even_list_elements(input_list):
    """ Use a list comprehension to return a new list that has 
        only the even elements of input_list in it.
    """
    return [x for x in input_list if x % 2 == 0]


def list_overlap_comp(list1, list2):
    """ Use a list comprehension to return a list that contains 
        only the elements that are in common between list1 and list2.
    """ 
    return [x for x in list1 if x in list2]


def div7list():
    """ Use a list comprehension to return a list of all of the numbers 
        from 1-1000 that are divisible by 7.
    """
    return[x for x in range(1000) if x % 7 == 0]


def has3list():
    """ Use a list comprehension to return a list of the numbers from 
        1-1000 that have a 3 in them.
    """
    return[x for x in range(1000) if str(3) in str(x)]


def cube_triples(input_list):
    """ Use a list comprehension to return a list with the cubes
        of the numbers divisible by three in the input_list.
    """
    return [x * x * x for x in input_list if x % 3 == 0]


def remove_vowels(input_string):
    """ Use a list comprehension to remove all of the vowels in the 
        input string, and then return the new string.
    """
    return ''.join(x for x in str(input_string) if str(x)!="a" and str(x)!="e" and str(x)!="i" and str(x)!="o" and str(x)!="u")


def short_words(input_string):
    """ Use a list comprehension to return a list of all of the words 
        in the input string that are less than 4 letters.
    """
    input_string = input_string.replace(".", "")
    input_string = input_string.replace(",", "")
    input_string = input_string.replace(";", "")
    input_string = input_string.replace("!", "")
    input_string = input_string.replace("?", "")
    input_string = input_string.replace(":", "")
    input_string = input_string.split()
    return [str(x) for x in input_string if len(str(x)) < 4]


# Challenge problem for extra credit:

def div_1digit():
    """ Use a nested list comprehension to find all of the numbers from 
        1-1000 that are divisible by any single digit besides 1 (2-9).
    """
    ##[x for x in range(100) if x % 2 == 0 or  ]


# More practice with Dictionaries, Files, and Text!
# Implement the following functions:

def longest_sentence(text_file_name):
    """ Read from the text file, split the data into sentences,
        and return the longest sentence in the file.
    """
    f = open(text_file_name, "r")
    data = f.read()
    data = data.replace("?", ".")
    data = data.replace("!", ".")
    data = data.replace(";", ".")
    longest = max(data.split("."), key = len)  
    f.close()
    return longest      

def longest_word(text_file_name):
    """ Read from the text file, split the data into words,
        and return the longest word in the file.
    """
    f = open(text_file_name, "r")
    data = f.read()
    data = data.replace('?', '')
    data = data.replace('!', '')
    data = data.replace(';', '')
    data = data.replace(',', '')
    data = data.replace('/', '')
    data = data.replace(':', '')
    data = data.replace('(', '')
    data = data.replace(')', '')
    data = data.replace('.', '')
    data = data.replace('_', '')
    data = data.replace('"', '')
    data = data.replace('__', '')
    longest = max(data.split(), key = len) 
    f.close()
    return longest

def num_unique_words(text_file_name):
    """ Read from the text file, split the data into words,
        and return the number of unique words in the file.
        HINT: Use a set!
    """
    f = open(text_file_name, "r")
    data = f.read()
    data = data.replace('?', '')
    data = data.replace('!', '')
    data = data.replace(';', '')
    data = data.replace(',', '')
    data = data.replace('/', '')
    data = data.replace(':', '')
    data = data.replace('-', '')
    data = data.replace('(', '')
    data = data.replace(')', '')
    data = data.replace('.', '')
    data = data.replace('_', '')
    data = data.replace('"', '')
    data = data.replace('--', '')
    data = data.lower().split()
    wordSet = set(data)
    return len(wordSet)
    


def most_frequent_word(text_file_name):
    """ Read from the text file, split the data into words,
        and return a tuple with the most frequently occuring word 
        in the file and the count of the number of times it appeared.
    """
    f = open(text_file_name, "r")
    data = f.read()
    data = data.replace('?', '')
    data = data.replace('!', '')
    data = data.replace(';', '')
    data = data.replace(',', '')
    data = data.replace('/', '')
    data = data.replace(':', '')
    data = data.replace('-', '')
    data = data.replace('(', '')
    data = data.replace(')', '')
    data = data.replace('.', '')
    data = data.replace('_', '')
    data = data.replace('"', '')
    data.lower() 
    data = data.split()
    wordDict = dict()
    for x in data:
        if x in wordDict:
            wordDict[x] += 1
        else:
             wordDict[x] = 1
    frequentword = max(wordDict, key = lambda x: wordDict[x])
    return (frequentword, wordDict[frequentword])
        
    

## work on this
def date_decoder(date_input):
    """ Accept a date in the "dd-MMM-yy" format (ex: 17-MAR-85 ) and 
        return a tuple in the form ( year, month_number, day).
        Create and use a dictionary suitable for decoding month names 
        to numbers. 
    """
    dateList = date_input.lower().split("-")
    monthDict = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
                 "jul": 7, "aug": 8,  "sep":9, "oct": 10, "nov": 11, "dec":12}
    return (dateList[2], monthDict[dateList[1]], dateList[0])

    
    
##work on this
def isit_random(lowest, highest, num_tries):
    """ Create and return a dictionary that is a histogram of how many 
        times the random.randInt function returns each value in the 
        range from 'lowest' to 'highest'. Run the randInt function a
        total number of times equal to 'num_tries'.
    """
    randomDict = dict()
    num = 0
    for x in range(num_tries + 1):
        num = random.randint(lowest, highest)
        if num in randomDict:
            randomDict[num] += 1
        else:
            randomDict[num] = 1
    return randomDict
             


# Extra challenge problem: Surpassing Phrases!

"""
Surpassing words are English words for which the gap between each adjacent
pair of letters strictly increases. These gaps are computed without 
"wrapping around" from Z to A.

For example:  http://i.imgur.com/XKiCnUc.png

Write a function to determine whether an entire phrase passed into a 
function is made of surpassing words. You can assume that all words are 
made of only alphabetic characters, and are separated by whitespace. 
We will consider the empty string and a 1-character string to be valid 
surpassing phrases.

is_surpassing_phrase("superb subway") # => True
is_surpassing_phrase("excellent train") # => False
is_surpassing_phrase("porky hogs") # => True
is_surpassing_phrase("plump pigs") # => False
is_surpassing_phrase("turnip fields") # => True
is_surpassing_phrase("root vegetable lands") # => False
is_surpassing_phrase("a") # => True
is_surpassing_phrase("") # => True

You may find the Python functions `ord` (one-character string to integer 
ordinal) and `chr` (integer ordinal to one-character string) useful to 
solve this puzzle.

ord('a') # => 97
chr(97) # => 'a'
"""

# Using the 'words' file on haiku, which are surpassing words? As a sanity check, I expect ~1931 distinct surpassing words.

def is_surpassing_phrase(input_string):
    """ Returns true if every word in the input_string is a surpassing
        word, and false otherwise.
    """
    pass


# I have more funky challenge problems if you need them!


if __name__ == "__main__":
   print(__author__ + "'s results:")
   ## input_list = [1,2,3,4,5,6]
  ##  list1 = [1,3,5]
   ## list2 = [2,3, 8]
   ## date_input = "26-JUL-02"
    ##input_string = "hello world! what's up?"
    #print(list_overlap_comp(list1,list2))
    #print(even_list_elements(input_list))
   # print(has3list())
    #print(cube_triples(input_list))
    #print(remove_vowels(input_string))
    #print(short_words(input_string))
    ##print(longest_sentence("rj_prologue.txt"))
    ##print(longest_word("rj_prologue.txt"))
    ##print(num_unique_words("rj_prologue.txt"))
    ####print(date_decoder(date_input))
   ## print(isit_random(2, 4, 10))


""" This module has sample tests for the PythonAdvPractice_2019.py file 

"""
import math
f=0 
extra=0
num_exer=5

try:
    import PythonAdvPractice_2019 as pap

    listOdd = [ 1, 3, 5, 7 ]
    listEven = [ 2, 4, 6, 8 ]
    listMixed = [ 2, 3, 4 ]
    listdups1 = [ 6, 6, 7 ]
    listdups2 = [ 3, 3, 3, 2, 2, 1, 0]
    list3 = [ 4, 9, 2, 3, 3, 5]
    emptyList = []

    print("Testing with these lists:")
    print("listOdd = " + str(listOdd))
    print("listEven = " + str(listEven))
    print("listMixed = " + str(listMixed))
    print("listdups1 = " + str(listdups1))
    print("listdups2 = " + str(listdups2))
    print("list3 = " + str(list3))
    print("emptyList = " + str(emptyList))

# def even_list_elements(input_list):
    """ Use a list comprehension/generator to return a new list that has 
        only the even elements of input_list in it.
    """
    try:
        test1 = pap.even_list_elements(listEven)
        test2 = pap.even_list_elements(listOdd)
        test3 = pap.even_list_elements(listMixed)
        test4 = pap.even_list_elements(listdups1)
        test5 = pap.even_list_elements(emptyList)
        if len(test1) != 4:
            print("FAILED: even_list_elements(listEven) returned: " + str(test1))
            f += 0.2
        else:
            print("passed: even_list_elements(listEven) with: " + str(test1))
        if len(test2) != 0:
            print("FAILED: even_list_elements(listOdd) returned: " + str(test2))
            f += 0.2
        else:
            print("passed: even_list_elements(listOdd) with: " + str(test2))
        if len(test3) != 2:
            print("FAILED: even_list_elements(listMixed) returned: " + str(test3))
            f += 0.2
        else:
            print("passed: even_list_elements(listMixed) with: " + str(test3))
        if len(test4) != 2:
            print("FAILED: even_list_elements(listdups1) returned: " + str(test4))
            f += 0.2
        else:
            print("passed: even_list_elements(listdups1) with: " + str(test4))
        if len(test5) != 0:
            print("FAILED: even_list_elements(emptyList) returned: " + str(test5))
            f += 0.2
        else:
            print("passed: even_list_elements(emptyList) with: " + str(test5))
    except Exception as ex:
        print(ex)
        print("FAILED: even_list_elements threw an exception.")
        f += 1


# def list_overlap_comp(list1, list2):
    """ Use a list comprehension/generator to return a list that contains 
        only the elements that are in common between list1 and list2.
    """ 

    try:
        test1 = pap.list_overlap_comp(listOdd, listMixed)
        test2 = pap.list_overlap_comp(listEven, listMixed)
        test3 = pap.list_overlap_comp(listOdd, listEven)
        test4 = pap.list_overlap_comp(listOdd, emptyList)
        test5 = pap.list_overlap_comp(listOdd, listdups2)
        if len(test1) != 1:
            print("FAILED: list_overlap_comp(listOdd,listMixed) returned: " + str(test1))
            f += 0.2
        else:
            print("passed: list_overlap_comp(listOdd,listMixed) with: " + str(test1))
        if len(test2) != 2:
            print("FAILED: list_overlap_comp(listEven,listMixed) returned: " + str(test2))
            f += 0.2
        else:
            print("passed: list_overlap_comp(listEven,listMixed) with: " + str(test2))
        if len(test3) != 0:
            print("FAILED: list_overlap_comp(listOdd,listEven) returned: " + str(test3))
            f += 0.2
        else:
            print("passed: list_overlap_comp(listOdd,listEven) with: " + str(test3))
        if len(test4) != 0:
            print("FAILED: list_overlap_comp(listOdd,emptyList) returned: " + str(test4))
            f += 0.2
        else:
            print("passed: list_overlap_comp(listOdd,emptyList) with: " + str(test4))
        if len(test5) != 2:
            print("FAILED: list_overlap_comp(listOdd,listdups2) returned: " + str(test5))
            f += 0.2
        else:
            print("passed: list_overlap_comp(listOdd,listdups2) with: " + str(test5))
    except Exception as ex:
        print(ex)
        print("FAILED: list_overlap_comp threw an exception.")
        f += 1

# More practice with Dictionaries, Files, and Text!
# Implement the following functions:

    test_files = []
    test_files.append("rj_prologue.txt")
    test_files.append("permutation.txt")
    test_files.append("UncannyValley.txt")
    
# def longest_sentence(text_file_name):
    """ Read from the text file, split the data into sentences,
        and return the longest sentence in the file.
    """
    answers = []
    answers.append("The fearful passage of their death-mark'd love,\nAnd the continuance of their parents' rage,\nWhich, but their children's end, nought could remove,\nIs now the two hours' traffic of our stage")
    answers.append("Objects out of sight didn't \"vanish\" entirely, if they influenced the ambient light, but Paul knew that the calculations would rarely be pursued beyond the crudest first-order approximations: Bosch's Garden of Earthly Delights reduced to an average reflectance value, a single grey rectangle - because once his back was turned, any more detail would have been wasted")
    answers.append("Later, in a room of his own, his bed had come with hollow metal posts whose plastic caps were easily removed, allowing him to toss in chewed pencil stubs, pins that had held newly bought school shirts elaborately folded around cardboard packaging, tacks that he'd bent out of shape with misaligned hammer blows while trying to form pictures in zinc on lumps of firewood, pieces of gravel that had made their way into his shoes, dried snot scraped from his handkerchief, and tiny, balled-up scraps of paper, each bearing a four- or five-word account of whatever seemed important at the time, building up a record of his life like a core sample slicing through geological strata, a find for future archaeologists far more exciting than any diary")
    try:
        for i in range(len(test_files)):
            output = pap.longest_sentence(test_files[i])
            if  output.strip().lower().rstrip('.!?;') != answers[i].strip().lower().rstrip('.!?;'):
                print("FAILED: longest_sentence(" + test_files[i] + ") returned: \n" + str(output) + "\n instead of: \n" + str(answers[i]))
                f += 0.333333333
            else:
                print("passed: longest_sentence(" + test_files[i] + ") with: \n" + str(output))
    except Exception as ex:
        print(ex)
        print("FAILED: longest_sentence threw an exception.")
        f += 1


# def longest_word(text_file_name):
    """ Read from the text file, split the data into words,
        and return the longest word in the file.
    """
    answers = []
    answers.append("misadventured")
    answers.append("soon-to-be-forgotten")
    answers.append("jurisprudentially")

    try:
        for i in range(len(test_files)):
            output = pap.longest_word(test_files[i])
            if len(output) != len(answers[i]):
                print("FAILED: longest_word(" + test_files[i] + ") returned: " + str(output) + " instead of: " + str(answers[i]))
                f += 0.333333333
            else:
                print("passed: longest_word(" + test_files[i] + ") with: " + str(output))
    except Exception as ex:
        print(ex)
        print("FAILED: longest_word threw an exception.")
        f += 1

# def num_unique_words(text_file_name):
    """ Read from the text file, split the data into words,
        and return the number of unique words in the file.
        HINT: Use a set!
    """
    answers = []
    answers.append(80)
    answers.append(1540)
    answers.append(2962)

    try:
        for i in range(len(test_files)):
            output = pap.num_unique_words(test_files[i])
            if math.fabs(output - answers[i]) > max(2,answers[i]/100):
                print("FAILED: num_unique_words(" + test_files[i] + ") returned: " + str(output) + " instead of: " + str(answers[i]))
                f += 0.333333333
            else:
                print("passed: num_unique_words(" + test_files[i] + ") with: " + str(output))
    except Exception as ex:
        print(ex)
        print("FAILED: num_unique_words threw an exception.")
        f += 1

# def most_frequent_word(text_file_name):
    """ Read from the text file, split the data into words,
        and return a tuple with the most frequently occuring word 
        in the file and the count of the number of times it apapared.
    """
    answers = []
    answers.append(('their',6))
    answers.append(('the', 266))
    answers.append(('the', 720))
    try:
        for i in range(len(test_files)):
            output = pap.most_frequent_word(test_files[i])
            if  output[0].lower() != answers[i][0].lower() and math.fabs(output[1] - answers[i][1]) > max(2,answers[i][1]/100) :
                print("FAILED: most_frequent_word(" + test_files[i] + ") returned: " + str(output) + " instead of: " + str(answers[i]))
                f += 0.333333333
            else:
                print("passed: most_frequent_word(" + test_files[i] + ") with: " + str(output))
    except Exception as ex:
        print(ex)
        print("FAILED: most_frequent_word threw an exception.")
        f += 1

except Exception as ex:
    print(ex)
    print("FAILED: PythonAdvPractice2019.py file does not execute at all, or this file was not implemented.")
    f = 2

print("\n")
print("SUMMARY:")
print("Passed " + str(round(num_exer-f,2)) + " out of " + str(num_exer) + " exercises.")

 
"""
"""