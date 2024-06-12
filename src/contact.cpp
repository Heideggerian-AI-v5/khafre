#define PY_SSIZE_T_CLEAN
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
#if 0
    float **crdepth = depth;
    uint8_t **crdil=dil;
    uint8_t **crmaskN=maskN;
    uint8_t **crmaskD=maskD;
    uint8_t **crcontP=contP;
    int lastSearch = searchWidth+1;
    for(int i=0; i < rows; i++)
    {
        float *cdepth = *crdepth;
        uint8_t *cdil=*crdil;
        uint8_t *cmaskN=*crmaskN;
        //uint8_t *cmaskD=*crmaskD;
        uint8_t *ccontP=*crcontP;
        for(int j=0; j < cols; j++)
        {
            if((0!=*cdil) && (0!=*cmaskN))
            {
                uint8_t found = 0;
                int k = -searchWidth;
                float **cordepth = crdepth+k;
                uint8_t **cormaskD = crmaskD+k;
                while((0==found) && (k < lastSearch))
                {
                    int l = -searchWidth;
                    float *codepth = (*cordepth)+j+l;
                    uint8_t *comaskD = (*cormaskD)+j+l;
                    while((0==found) && (l < lastSearch))
                    {
                        int cr = i+k;
                        int cc = j+l;
                        if((0<=cr) && (cr<rows) && (0<=cc) && (cc<cols) && (0!=*comaskD) && close(*cdepth, *codepth, threshold))
                        {
                            found = 255;
                        }
                        l++;
                        codepth++;
                        comaskD++;
                    }
                    k++;
                    cordepth++;
                    cormaskD++;
                }
                if(0 != found)
                {
                    *ccontP = 255;
                }
            }
            cdepth++;
            cdil++;
            cmaskN++;
            //cmaskD++;
            ccontP++;
        }
        crdepth++;
        crdil++;
        crmaskN++;
        crmaskD++;
        crcontP++;
    }
#endif
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

#if 0
static PyObject * contact_contact(PyObject *self, PyObject *args) {
    (void)self;
    int searchWidth;
    float threshold;
    int rows;
    int cols;
    uint8_t **dilS;
    uint8_t **dilO;
    uint8_t **maskS;
    uint8_t **maskO;
    float **depthImg;
    uint8_t **contPS;
    uint8_t **contPO;
    if (!PyArg_ParseTuple(args, "ifiiB**B**B**B**f**B**B**", &searchWidth, &threshold, &rows, &cols, &dilS, &dilO, &maskS, &maskO, &depthImg, &contPS, &contPO))
        return NULL;
    contactFn(searchWidth, threshold, rows, cols, dilS, dilO, maskS, maskO, depthImg, contPS, contPO);
    Py_RETURN_NONE;
}
#endif

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
    import_array();
    return PyModule_Create(&contactcpp);
}

#if 0
#include <stdio.h>
#include <stdint.h>

#if defined(_MSC_VER)
    //  Microsoft 
    #define EXPORT __declspec(dllexport)
    #define IMPORT __declspec(dllimport)
#elif defined(__GNUC__)
    //  GCC
    #define EXPORT __attribute__((visibility("default")))
    #define IMPORT
#else
    //  do nothing and hope for the best?
    #define EXPORT
    #define IMPORT
    #pragma warning Unknown dynamic link import/export semantics.
#endif

extern "C"
{
    EXPORT void contact(int, float, int, int, uint8_t **, uint8_t **, uint8_t **, uint8_t **, float **, uint8_t **, uint8_t **);
}

uint8_t close(float a, float b, float threshold)
{
    float x = a-b;
    if(x < 0)
    {
        x = -x;
    }
    return (x<threshold);
}

void contactDM(int searchWidth, float threshold, int rows, int cols, uint8_t **dil, uint8_t **maskN, uint8_t **maskD, float **depth, uint8_t **contP)
{
    float **crdepth = depth;
    uint8_t **crdil=dil;
    uint8_t **crmaskN=maskN;
    uint8_t **crmaskD=maskD;
    uint8_t **crcontP=contP;
    int lastSearch = searchWidth+1;
    for(int i=0; i < rows; i++)
    {
        float *cdepth = *crdepth;
        uint8_t *cdil=*crdil;
        uint8_t *cmaskN=*crmaskN;
        //uint8_t *cmaskD=*crmaskD;
        uint8_t *ccontP=*crcontP;
        for(int j=0; j < cols; j++)
        {
            if((0!=*cdil) && (0!=*cmaskN))
            {
                uint8_t found = 0;
                int k = -searchWidth;
                float **cordepth = crdepth+k;
                uint8_t **cormaskD = crmaskD+k;
                while((0==found) && (k < lastSearch))
                {
                    int l = -searchWidth;
                    float *codepth = (*cordepth)+j+l;
                    uint8_t *comaskD = (*cormaskD)+j+l;
                    while((0==found) && (l < lastSearch))
                    {
                        int cr = i+k;
                        int cc = j+l;
                        if((0<=cr) && (cr<rows) && (0<=cc) && (cc<cols) && (0!=*comaskD) && close(*cdepth, *codepth, threshold))
                        {
                            found = 255;
                        }
                        l++;
                        codepth++;
                        comaskD++;
                    }
                    k++;
                    cordepth++;
                    cormaskD++;
                }
                if(0 != found)
                {
                    *ccontP = 255;
                }
            }
            cdepth++;
            cdil++;
            cmaskN++;
            //cmaskD++;
            ccontP++;
        }
        crdepth++;
        crdil++;
        crmaskN++;
        crmaskD++;
        crcontP++;
    }
}

void contact(int searchWidth, float threshold, int rows, int cols, uint8_t **dilS, uint8_t **dilO, uint8_t **maskS, uint8_t **maskO, float **depthImg, uint8_t **contPS, uint8_t **contPO)
{
    contactDM(searchWidth, threshold, rows, cols, dilS, maskO, maskS, depthImg, contPO);
    contactDM(searchWidth, threshold, rows, cols, dilO, maskS, maskO, depthImg, contPS);
}
#endif

