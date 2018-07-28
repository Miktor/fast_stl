#include <Python.h>
#include "c_loess.h"

/*
 * Implements an example function.
 */
PyDoc_STRVAR(fast_loess_example_doc, "example(obj, number)\
\
Example function");

PyObject *fast_loess_example(PyObject *self, PyObject *args, PyObject *kwargs) {
    /* Shared references that do not need Py_DECREF before returning. */
    PyObject *obj = NULL;
    int number = 0;

    /* Parse positional and keyword arguments */
    static char* keywords[] = { "obj", "number", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi", keywords, &obj, &number)) {
        return NULL;
    }

    /* Function implementation starts here */

    if (number < 0) {
        PyErr_SetObject(PyExc_ValueError, obj);
        return NULL;    /* return NULL indicates error */
    }

    Py_RETURN_NONE;
}

/*
 * List of functions to add to fast_loess in exec_fast_loess().
 */
static PyMethodDef fast_loess_functions[] = {
    { "example", (PyCFunction)fast_loess_example, METH_VARARGS | METH_KEYWORDS, fast_loess_example_doc },
    { NULL, NULL, 0, NULL } /* marks end of array */
};

/*
 * Initialize fast_loess. May be called multiple times, so avoid
 * using static state.
 */
int exec_fast_loess(PyObject *module) {
    PyModule_AddFunctions(module, fast_loess_functions);

    PyModule_AddStringConstant(module, "__author__", "Miktor");
    PyModule_AddStringConstant(module, "__version__", "1.0.0");
    PyModule_AddIntConstant(module, "year", 2018);

    return 0; /* success */
}

/*
 * Documentation for fast_loess.
 */
PyDoc_STRVAR(fast_loess_doc, "The fast_loess module");


static PyModuleDef_Slot fast_loess_slots[] = {
    { Py_mod_exec, exec_fast_loess },
    { 0, NULL }
};

static PyModuleDef fast_loess_def = {
    PyModuleDef_HEAD_INIT,
    "fast_loess",
    fast_loess_doc,
    0,              /* m_size */
    NULL,           /* m_methods */
    fast_loess_slots,
    NULL,           /* m_traverse */
    NULL,           /* m_clear */
    NULL,           /* m_free */
};

PyMODINIT_FUNC PyInit_fast_loess() {
    return PyModuleDef_Init(&fast_loess_def);
}
