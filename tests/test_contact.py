import khafre.contact
import numpy
import pytest

def test_contact_searchWidth_1():
    a=numpy.array([[0,1,1,1,0],[0,1,1,1,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]],dtype=numpy.uint8)
    b=numpy.array([[0,0,0,0,0],[0,0,0,0,0],[0,1,1,1,0],[0,1,1,1,0],[0,0,0,0,0]],dtype=numpy.uint8)
    aD=numpy.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[0,0,0,0,0],[0,0,0,0,0]],dtype=numpy.uint8)
    bD=numpy.array([[0,0,0,0,0],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]],dtype=numpy.uint8)
    maskImgs={"a":a,"b":b}
    dilImgs={"a":aD,"b":bD}
    d=numpy.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]],dtype=numpy.float32)
    aC,bC=khafre.contact.contact(1,0.1,5,5,"a","b",dilImgs,maskImgs,d,erodeMasks=False)
    expectedA=numpy.array([[0,0,0,0,0],[0,255,255,255,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]],dtype=numpy.uint8)
    expectedB=numpy.array([[0,0,0,0,0],[0,0,0,0,0],[0,255,255,255,0],[0,0,0,0,0],[0,0,0,0,0]],dtype=numpy.uint8)
    assert(numpy.array_equal(aC,expectedA))
    assert(numpy.array_equal(bC,expectedB))

def test_contact_searchWidth_3_nogap():
    a=numpy.array([[0,1,1,1,0],[0,1,1,1,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]],dtype=numpy.uint8)
    b=numpy.array([[0,0,0,0,0],[0,0,0,0,0],[0,1,1,1,0],[0,1,1,1,0],[0,0,0,0,0]],dtype=numpy.uint8)
    aD=numpy.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[0,0,0,0,0],[0,0,0,0,0]],dtype=numpy.uint8)
    bD=numpy.array([[0,0,0,0,0],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]],dtype=numpy.uint8)
    maskImgs={"a":a,"b":b}
    dilImgs={"a":aD,"b":bD}
    d=numpy.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]],dtype=numpy.float32)
    aC,bC=khafre.contact.contact(3,0.1,5,5,"a","b",dilImgs,maskImgs,d,erodeMasks=False)
    expectedA=numpy.array([[0,0,0,0,0],[0,255,255,255,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]],dtype=numpy.uint8)
    expectedB=numpy.array([[0,0,0,0,0],[0,0,0,0,0],[0,255,255,255,0],[0,0,0,0,0],[0,0,0,0,0]],dtype=numpy.uint8)
    assert(numpy.array_equal(aC,expectedA))
    assert(numpy.array_equal(bC,expectedB))

def test_contact_searchWidth_3_gap():
    a=numpy.array([[0,1,1,1,0],[0,1,1,1,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]],dtype=numpy.uint8)
    b=numpy.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,1,1,1,0],[0,1,1,1,0]],dtype=numpy.uint8)
    aD=numpy.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[0,0,0,0,0]],dtype=numpy.uint8)
    bD=numpy.array([[0,0,0,0,0],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]],dtype=numpy.uint8)
    maskImgs={"a":a,"b":b}
    dilImgs={"a":aD,"b":bD}
    d=numpy.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]],dtype=numpy.float32)
    aC,bC=khafre.contact.contact(3,0.1,5,5,"a","b",dilImgs,maskImgs,d,erodeMasks=False)
    expectedA=numpy.array([[0,0,0,0,0],[0,255,255,255,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]],dtype=numpy.uint8)
    expectedB=numpy.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,255,255,255,0],[0,0,0,0,0]],dtype=numpy.uint8)
    assert(numpy.array_equal(aC,expectedA))
    assert(numpy.array_equal(bC,expectedB))

