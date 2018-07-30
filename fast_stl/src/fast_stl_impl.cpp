#include <Python.h>
#include <utility>
#include <numpy\ndarraytypes.h>
#include <numpy\arrayobject.h>
#include <numpy\npy_math.h>
#include "loess.h"

namespace
{
	template <NPY_TYPES Type>
	struct PythonTypeHelper {};

	template <> struct PythonTypeHelper<NPY_FLOAT>		{ typedef npy_float Type; };
	template <> struct PythonTypeHelper<NPY_DOUBLE>		{ typedef npy_double Type; };
	template <> struct PythonTypeHelper<NPY_LONG>		{ typedef npy_long Type; };
	template <> struct PythonTypeHelper<NPY_ULONG>		{ typedef npy_ulong Type; };
	template <> struct PythonTypeHelper<NPY_LONGLONG>	{ typedef npy_longlong Type; };
	template <> struct PythonTypeHelper<NPY_ULONGLONG>	{ typedef npy_ulonglong Type; };

	template <NPY_TYPES NpType>
	bool loess_type_helper(PyObject *soretd_x_array, const uint32_t soretd_x_length,
					  PyObject *soretd_y_array,
					  PyObject *samples_array, const uint32_t samples_length,
					  const uint32_t neighbours,
					  PyObject *out_array)
	{
		typedef PythonTypeHelper<NpType>::Type Type;

		Type *out_data = (Type*)PyArray_DATA(out_array);
		Type *soretd_x_array_data = (Type*)PyArray_DATA(soretd_x_array);
		Type *soretd_y_array_data = (Type*)PyArray_DATA(soretd_y_array);
		Type *samples_array_data = (Type*)PyArray_DATA(samples_array);

		return loess<Type>(soretd_x_array_data, soretd_x_length, soretd_y_array_data, samples_array_data, samples_length, neighbours, out_data);
	}

	bool loess_helper(NPY_TYPES array_type, PyObject *soretd_x_array, const uint32_t soretd_x_length,
					  PyObject *soretd_y_array,
					  PyObject *samples_array, const uint32_t samples_length,
					  const uint32_t neighbours,
					  PyObject *out_array)
	{
		bool result = false;
		switch(array_type)
		{
		case NPY_LONG:
			result = loess_type_helper<NPY_ULONG>(soretd_x_array, soretd_x_length, soretd_y_array, samples_array, samples_length, neighbours, out_array);
			break;
		case NPY_ULONG:
			result = loess_type_helper<NPY_ULONG>(soretd_x_array, soretd_x_length, soretd_y_array, samples_array, samples_length, neighbours, out_array);
			break;
		case NPY_LONGLONG:
			result = loess_type_helper<NPY_LONGLONG>(soretd_x_array, soretd_x_length, soretd_y_array, samples_array, samples_length, neighbours, out_array);
			break;
		case NPY_ULONGLONG:
			result = loess_type_helper<NPY_ULONGLONG>(soretd_x_array, soretd_x_length, soretd_y_array, samples_array, samples_length, neighbours, out_array);
			break;
		case NPY_FLOAT:
			result = loess_type_helper<NPY_FLOAT>(soretd_x_array, soretd_x_length, soretd_y_array, samples_array, samples_length, neighbours, out_array);
			break;
		case NPY_DOUBLE:
			result = loess_type_helper<NPY_DOUBLE>(soretd_x_array, soretd_x_length, soretd_y_array, samples_array, samples_length, neighbours, out_array);
			break;
		default:
			break;
		}
		return result;
	}
}
/*
	* Implements an example function.
	*/
PyDoc_STRVAR(fast_loess_doc, "loess(soretd_x, soretd_y, samples, neighbours)\
\
Example function");

PyObject *fast_loess(PyObject *self, PyObject *args)
{
	/* Shared references that do not need Py_DECREF before returning. */
	int neighbours = 0;
	PyObject *soretd_x = NULL;
	PyObject *soretd_y = NULL;
	PyObject *samples = NULL;

	/* Parse positional and keyword arguments */
	static char* keywords[] = {"soretd_x", "soretd_y", "samples", "neighbours", NULL};
	if(!PyArg_ParseTuple(args, "OOOI", &soretd_x, &soretd_y, &samples, &neighbours))
	{
		return NULL;
	}

	struct DataHelper
	{
		PyObject *soretd_x = NULL;
		PyObject *soretd_y = NULL;
		PyObject *samples = NULL;

		PyObject *out_array = NULL;

		~DataHelper()
		{
			Py_XDECREF(soretd_x);
			Py_XDECREF(soretd_y);
			Py_XDECREF(samples);
			Py_XDECREF(out_array);
		}
	} data_helper{soretd_x, soretd_y, samples, nullptr};

	if(soretd_x == NULL || soretd_y == NULL || samples == NULL || neighbours < 1)
	{
		PyErr_SetString(PyExc_TypeError, "Couldn't parse the input arrays.");
		return NULL;
	}

	if(PyArray_API == NULL)
		import_array();

	PyObject *soretd_x_array = PyArray_FROM_OF(soretd_x, NPY_IN_ARRAY);
	PyObject *soretd_y_array = PyArray_FROM_OF(soretd_y, NPY_IN_ARRAY);
	PyObject *samples_array = PyArray_FROM_OF(samples, NPY_IN_ARRAY);
	if(soretd_x_array == NULL || soretd_y_array == NULL || samples_array == NULL)
	{
		PyErr_SetString(PyExc_TypeError, "Couldn't parse the input arrays.");
		return NULL;
	}

	const NPY_TYPES array_type = static_cast<NPY_TYPES>(PyArray_TYPE(soretd_x_array));
	if(!(array_type == NPY_LONG ||
		 array_type == NPY_ULONG ||
		 array_type == NPY_LONGLONG ||
		 array_type == NPY_ULONGLONG ||
		 array_type == NPY_FLOAT ||
		 array_type == NPY_DOUBLE))
	{
		PyErr_SetString(PyExc_TypeError, "Invalid array types.");
		return NULL;
	}

	/* Check that the input arrays are 1D. */
	int soretd_x_dim = (int)PyArray_NDIM(soretd_x_array);
	int soretd_y_dim = (int)PyArray_NDIM(soretd_y_array);
	int samples_dim = (int)PyArray_NDIM(samples_array);
	if(soretd_x_dim != 1 || soretd_y_dim != 1 || samples_dim != 1)
	{
		PyErr_SetString(PyExc_TypeError, "The input arrays must be 1D.");
		return NULL;
	}

	/* Get the lengths of the outputs. */
	int soretd_x_length = (int)PyArray_DIM(soretd_x_array, 0);
	int soretd_y_length = (int)PyArray_DIM(soretd_y_array, 0);
	if(soretd_x_length != soretd_y_length)
	{
		PyErr_SetString(PyExc_TypeError, "sorted_x and sorted_y must be same length.");
		return NULL;
	}

	int samples_length = (int)PyArray_DIM(samples_array, 0);
	if(samples_length < 1)
	{
		PyErr_SetString(PyExc_TypeError, "samples should be not empty.");
		return NULL;
	}

	/* Build the output array. */
	npy_intp dims{samples_length};
	PyObject *out_array = PyArray_SimpleNew(1, &dims, array_type);
	if(out_array == NULL)
	{
		PyErr_SetString(PyExc_RuntimeError, "Couldn't build output array.");
		return NULL;
	}		
	data_helper.out_array = out_array;

	if(!loess_helper(array_type, soretd_x_array, soretd_x_length, soretd_y_array, samples_array, samples_length, neighbours, out_array))
	{
		PyErr_SetString(PyExc_RuntimeError, "Failed to calculate loess.");
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

PyMODINIT_FUNC PyInit_fast_stl_impl()
{
	return PyModuleDef_Init(&fast_stl_impl_def);
}
