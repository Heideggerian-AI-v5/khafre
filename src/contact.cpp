#define PY_SSIZE_T_CLEAN
#include <cassert>
#include <Python.h>
#include "numpy/arrayobject.h"
#include <iostream>

uint8_t close(float a, float b, float threshold)
{
    float x = a-b;
    if(x < 0)
    {
        x = -x;
    }
    return (x<threshold);
}

void contactDM(int searchWidth, float threshold, int rows, int cols, PyArrayObject *dil, PyArrayObject *maskN, PyArrayObject *maskD, PyArrayObject *depth, PyArrayObject *contP)
{
    if(searchWidth>cols-1)
        searchWidth=cols-1;
    if(searchWidth>rows-1)
        searchWidth=rows-1;
    int lastSearch = searchWidth+1;
    int windowSize = lastSearch+searchWidth;
    int strideUp = -(cols*searchWidth);
    int strideLeft = -searchWidth;
    uint8_t *dil_Element = (uint8_t*)PyArray_DATA(dil);
    uint8_t *maskN_Element = (uint8_t*)PyArray_DATA(maskN);
    uint8_t *maskD_Element = (uint8_t*)PyArray_DATA(maskD);
    float *depth_Element = (float*)PyArray_DATA(depth);
    uint8_t *contP_Element = (uint8_t*)PyArray_DATA(contP);
    for(int i=0; i<rows; i++) // Loop over rows of the dilated object (dil) and mask for the normal object (maskN)
    {
        for(int j=0; j<cols; j++) // Loop over row elements of a row of the dilated object (dil) and mask for the normal object (maskN)
        {
            if((0!=*dil_Element) && (0!=*maskN_Element)) // Is this a point where the dilated object and the normal object mask overlap?
            {
                uint8_t found = 0;
                int k = -searchWidth;
                float* codepth_Element = depth_Element + strideUp + strideLeft; // initialize pointer to upper left corner of searched area
                uint8_t *comaskD_Element = maskD_Element + strideUp + strideLeft;
                while((0==found) && (k<lastSearch)) // Look in an area of size (2*searchWidth+1)^2 around this position
                {
                    int cr = i+k;
                    int l = -searchWidth;
                    while((0==found) && (l<lastSearch))
                    {
                        int cc = j+l;
                        if((0<=cr) && (cr<rows) && (0<=cc) && (cc<cols) && (0!=*comaskD_Element) && close(*depth_Element, *codepth_Element, threshold))
                            found = 255;
                        l++;
                        codepth_Element++;
                        comaskD_Element++;
                    }
                    k++;
                    codepth_Element+=-windowSize+cols; // reset pointer position to the start of the window searched in the current row, then go to the next row
                    comaskD_Element+=-windowSize+cols;
                }
                if(0 != found)
                    *contP_Element = 255;
            }
            dil_Element++;
            maskN_Element++;
            maskD_Element++;
            depth_Element++;
            contP_Element++;
        }
    }
}

/* Docstring for our Python module. */
PyDoc_STRVAR(
    docstring,
    "Compute contact masks based on an image and a depth image.");

inline PyArrayObject* _parse2D(PyObject*arg, NPY_TYPES DTYPE)
{
    return (PyArrayObject*)PyArray_FROM_OTF(arg,PyArray_ObjectType(arg,DTYPE), NPY_ARRAY_IN_ARRAY);
}

static PyObject* contact_contact(PyObject *self, PyObject *args)
{
    (void)self;
    int searchWidth;
    float threshold;
    int rows, cols;
    PyObject * argdilS=NULL; // Regular Python/C API
    PyArrayObject * dilS=NULL; // Extended Numpy/C API
    PyObject * argdilO=NULL; // Regular Python/C API
    PyArrayObject * dilO=NULL; // Extended Numpy/C API
    PyObject * argmaskS=NULL; // Regular Python/C API
    PyArrayObject * maskS=NULL; // Extended Numpy/C API
    PyObject * argmaskO=NULL; // Regular Python/C API
    PyArrayObject * maskO=NULL; // Extended Numpy/C API
    PyObject * argdepthImg=NULL; // Regular Python/C API
    PyArrayObject * depthImg=NULL; // Extended Numpy/C API
    PyObject * argcontPS=NULL; // Regular Python/C API
    PyArrayObject * contPS=NULL; // Extended Numpy/C API
    PyObject * argcontPO=NULL; // Regular Python/C API
    PyArrayObject * contPO=NULL; // Extended Numpy/C API
    if (!PyArg_ParseTuple(args, "ifiiOOOOOOO", &searchWidth, &threshold, &rows, &cols, &argdilS, &argdilO, &argmaskS, &argmaskO, &argdepthImg, &argcontPS, &argcontPO))
    {
        PyErr_SetString(PyExc_ValueError, "Error parsing arguments.");
        return NULL;
    }
    dilS = _parse2D(argdilS, NPY_UINT8);
    dilO = _parse2D(argdilO, NPY_UINT8);
    maskS = _parse2D(argmaskS, NPY_UINT8);
    maskO = _parse2D(argmaskO, NPY_UINT8);
    contPS = _parse2D(argcontPS, NPY_UINT8);
    contPO = _parse2D(argcontPO, NPY_UINT8);
    depthImg = _parse2D(argdepthImg, NPY_FLOAT);
    
    if((NULL==dilS)&&(NULL==dilO)&&(NULL==maskS)&&(NULL==maskO)&&(NULL==contPS)&&(NULL==contPO)&&(NULL==depthImg))
        Py_RETURN_NONE;
    contactDM(searchWidth, threshold, rows, cols, dilS, maskO, maskS, depthImg, contPO);
    contactDM(searchWidth, threshold, rows, cols, dilO, maskS, maskO, depthImg, contPS);
    Py_RETURN_NONE;
}

/* Define the functions/methods that this module exports. */
static PyMethodDef AddModuleMethods[] = {
    {"_contact", contact_contact, METH_VARARGS, "Compute contact masks."},
    {NULL, NULL, 0, NULL}, /* Sentinel */
};

/* Define the actual module. */
static struct PyModuleDef contactcpp = {
    PyModuleDef_HEAD_INIT,
    "contactcpp", /* name of module */
    docstring,     /* module documentation, may be NULL */
    -1,            /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    AddModuleMethods,
    NULL, //m_slots
    NULL, //m_traverse
    NULL, //m_clear
    NULL, //m_free
};

/* The main entry point that is called by Python when our module is imported. */
PyMODINIT_FUNC PyInit_contactcpp(void) {
    assert(CHAR_BIT == 8); // Require 8bit chars
    assert(sizeof(float) == 4); // Require 32bit floats
    import_array();
    return PyModule_Create(&contactcpp);
}

