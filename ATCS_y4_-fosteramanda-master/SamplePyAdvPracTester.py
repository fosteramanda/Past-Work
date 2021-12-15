""" This module has sample tests for the PythonAdvPractice_2019.py file 
"""
import math
f=0 
extra=0

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

except Exception as ex:
    print(ex)
    print("FAILED: PythonAdvPractice2019.py file does not execute at all, or this file was not implemented.")
    f = 2

print("\n")
print("SUMMARY:")
print("Passed " + str(round(2-f,2)) + " out of 2 exercises.")
print("Earned " + str(round(extra,2)) + " extra credits.")

