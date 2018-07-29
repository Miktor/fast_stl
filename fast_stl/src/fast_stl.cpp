#include <Python.h>
#include <numpy\ndarraytypes.h>
#include <numpy\arrayobject.h>
#include <numpy\npy_math.h>
#include "loess.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
	* Implements an example function.
	*/
PyDoc_STRVAR(fast_loess_doc, "loess(soretd_x, soretd_y, samples, neighbours)\
\
Example function");

	PyObject *fast_loess(PyObject *self, PyObject *args)
	{
		/* Shared references that do not need Py_DECREF before returning. */
		PyObject *soretd_x = NULL;
		PyObject *soretd_y = NULL;
		PyObject *samples = NULL;
		int neighbours = 0;

		/* Parse positional and keyword arguments */
		static char* keywords[] = {"soretd_x", "soretd_y", "samples", "neighbours", NULL};
		if(!PyArg_ParseTuple(args, "OOOI", &soretd_x, &soretd_y, &samples, &neighbours))
		{
			return NULL;
		}

		if(soretd_x == NULL || soretd_y == NULL || samples == NULL || neighbours < 1)
		{
			PyErr_SetString(PyExc_TypeError, "Couldn't parse the input arrays.");
			Py_XDECREF(soretd_x);
			Py_XDECREF(soretd_y);
			Py_XDECREF(samples);
			return NULL;
		}

		if(PyArray_API == NULL)
			import_array();

		PyObject *soretd_x_array = PyArray_FROM_OTF(soretd_x, NPY_FLOAT, NPY_IN_ARRAY);
		PyObject *soretd_y_array = PyArray_FROM_OTF(soretd_y, NPY_FLOAT, NPY_IN_ARRAY);
		PyObject *samples_array = PyArray_FROM_OTF(samples, NPY_FLOAT, NPY_IN_ARRAY);

		if(soretd_x_array == NULL || soretd_y_array == NULL || samples_array == NULL)
		{
			PyErr_SetString(PyExc_TypeError, "Couldn't parse the input arrays.");
			Py_XDECREF(soretd_x_array);
			Py_XDECREF(soretd_y_array);
			Py_XDECREF(samples_array);
			return NULL;
		}

		/* Check that the input arrays are 1D. */
		int soretd_x_dim = (int)PyArray_NDIM(soretd_x_array);
		int soretd_y_dim = (int)PyArray_NDIM(soretd_y_array);
		int samples_dim = (int)PyArray_NDIM(samples_array);
		if(soretd_x_dim != 1 || soretd_y_dim != 1 || samples_dim != 1)
		{
			PyErr_SetString(PyExc_TypeError, "The input arrays must be 1D.");
			Py_XDECREF(soretd_x_array);
			Py_XDECREF(soretd_y_array);
			Py_XDECREF(samples_array);
			return NULL;
		}

		/* Get the lengths of the outputs. */
		int soretd_x_length = (int)PyArray_DIM(soretd_x_array, 0);
		int soretd_y_length = (int)PyArray_DIM(soretd_y_array, 0);
		if(soretd_x_length != soretd_y_length)
		{
			PyErr_SetString(PyExc_TypeError, "sorted_x and sorted_y must be same length.");
			Py_XDECREF(soretd_x_array);
			Py_XDECREF(soretd_y_array);
			Py_XDECREF(samples_array);
			return NULL;
		}

		int samples_length = (int)PyArray_DIM(samples_array, 0);
		if(samples_length < 1)
		{
			PyErr_SetString(PyExc_TypeError, "samples should be not empty.");
			Py_XDECREF(soretd_x_array);
			Py_XDECREF(soretd_y_array);
			Py_XDECREF(samples_array);
			return NULL;
		}

		/* Build the output array. */
		npy_intp dims{samples_length};
		PyObject *out_array = PyArray_SimpleNew(1, &dims, NPY_FLOAT);
		if(out_array == NULL)
		{
			PyErr_SetString(PyExc_RuntimeError, "Couldn't build output array.");
			Py_XDECREF(soretd_x_array);
			Py_XDECREF(soretd_y_array);
			Py_XDECREF(samples_array);
			Py_XDECREF(out_array);
			return NULL;
		}
		float *out_data = (float*)PyArray_DATA(out_array);

		float *soretd_x_array_data = (float*)PyArray_DATA(soretd_x_array);
		float *soretd_y_array_data = (float*)PyArray_DATA(soretd_y_array);
		float *samples_array_data = (float*)PyArray_DATA(soretd_y_array);

		/* Function implementation starts here */

		if(!loess(soretd_x_array_data, soretd_x_length, soretd_y_array_data, samples_array_data, samples_length, neighbours, out_data))
		{
			PyErr_SetString(PyExc_RuntimeError, "Failed to calculate loess.");
			Py_XDECREF(soretd_x_array);
			Py_XDECREF(soretd_y_array);
			Py_XDECREF(samples_array);
			Py_XDECREF(out_array);
			return NULL;    /* return NULL indicates error */
		}

		return out_array;
	}

	/*
	 * List of functions to add to fast_loess in exec_fast_loess().
	 */
	static PyMethodDef fast_loess_functions[] = {
		{ "loess", (PyCFunction)fast_loess, METH_VARARGS, fast_loess_doc },
		{ NULL, NULL, 0, NULL } /* marks end of array */
	};

	/*
	 * Initialize fast_loess. May be called multiple times, so avoid
	 * using static state.
	 */
	int exec_fast_loess(PyObject *module)
	{
		PyModule_AddFunctions(module, fast_loess_functions);

		PyModule_AddStringConstant(module, "__author__", "Dmitry Gladky");
		PyModule_AddStringConstant(module, "__version__", "0.0.1");
		PyModule_AddIntConstant(module, "year", 2018);

		return 0; /* success */
	}

	/*
	 * Documentation for fast_stl_impl.
	 */
	PyDoc_STRVAR(fast_stl_doc, "The fast_stl_impl module");


	static PyModuleDef_Slot fast_loess_slots[] = {
		{ Py_mod_exec, exec_fast_loess },
		{ 0, NULL }
	};

	static PyModuleDef fast_stl_impl_def = {
		PyModuleDef_HEAD_INIT,
		"fast_stl_impl",
		fast_stl_doc,
		0,              /* m_size */
		NULL,           /* m_methods */
		fast_loess_slots,
		NULL,           /* m_traverse */
		NULL,           /* m_clear */
		NULL,           /* m_free */
	};

	PyMODINIT_FUNC PyInit_fast_loess()
	{
		return PyModuleDef_Init(&fast_stl_impl_def);
	}
}