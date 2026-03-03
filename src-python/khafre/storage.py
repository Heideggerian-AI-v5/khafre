import ctypes
import cv2 as cv
from khafre.bricks import ReifiedProcess
from multiprocessing import Value
import mimetypes
import numpy
import os
import pickle
from PIL import Image
import socket
import struct
import time

class StoreTriples(ReifiedProcess):
    def __init__(self, predicateMap=None):
        super().__init__()
        self._folder = None
        self._collection = "triples"
        if predicateMap is None:
            predicateMap = {}
        self._predicateMap = predicateMap
        self._logPrefix = '''@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix xml: <http://www.w3.org/XML/1998/namespace> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix framester: <https://w3id.org/framester/schema/> .
@prefix dul: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#> .
@prefix dfl: <http://www.ease-crc.org/ont/SOMA_DFL.owl#> .
@prefix affordances_situations: <http://www.W3C.org/khafre/affordances_situations.owl#> .

@prefix log: <file://./log.owl#> .

        '''
    def _checkSubscriptionRequest(self, name, wire):
        return name in {"InpImg"}
    def _handleCommand(self, command):
        op, args = command
        if "SET_PATH" == op:
            self._folder = args[0]
        elif "SET_COLLECTION" == op:
            self._collection = args[0]
    def _doWork(self):
        if (self._folder is None):
            return
        notification, image, rate, dropped = self._requestSubscribedData("InpImg")
        print("Start store triples", self._collection, notification.get("imgId"), notification.get("triples"))
        fileName = notification.get("imgId")
        if fileName is None:
            return
        fileName = os.path.join(self._folder, str(fileName) + "_" + self._collection + ".ttl")
        with open(fileName, "w") as outfile:
            _ = outfile.write(self._logPrefix)
            _ = outfile.write("\n")
            for p, s, o in notification.get("triples", []):
                p = self._predicateMap.get(p, p)
                _ = outfile.write("%s %s %s .\n" % (s, p, o))
        print("End store triples", self._collection, notification.get("imgId"))
                

class StoreMasks(ReifiedProcess):
    def __init__(self):
        super().__init__()
        self._folder = None
        self._collection = "masks"
    def _checkSubscriptionRequest(self, name, wire):
        return name in {"InpImg"}
    def _handleCommand(self, command):
        op, args = command
        if "SET_PATH" == op:
            self._folder = args[0]
        elif "SET_COLLECTION" == op:
            self._collection = args[0]
    def _doWork(self):
        def _set2str(s):
            return '_'.join(sorted([str(x) for x in s]))
        if (self._folder is None):
            return
        notification, image, rate, dropped = self._requestSubscribedData("InpImg")
        fileName = notification.get("imgId")
        print("Start store mask", self._collection, notification.get("imgId"))
        height, width = image.shape[:2]
        if fileName is None:
            return
        fileName = os.path.join(self._folder, str(fileName) + "_" + self._collection + ".txt")
        with open(fileName, "w") as outfile:
            for segment in notification.get("segments", []):
                polygons = segment["polygons"]
                name = segment["name"]
                className = segment["type"]
                label = name
                for polygon in polygons:
                    pstr = ""
                    for p in polygon:
                        pstr += ("%f %f " % (p[0]/width, p[1]/height))
                    _ = outfile.write("%s %s\n" % (label, pstr))
        print("End store mask", self._collection, notification.get("imgId"))
