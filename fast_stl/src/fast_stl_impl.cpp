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
						   PyObject *out_array,
						   PyObject *opt_weights_scale)
	{
		typedef PythonTypeHelper<NpType>::Type Type;

		Type *out_data = (Type*)PyArray_DATA(out_array);
		Type *soretd_x_array_data = (Type*)PyArray_DATA(soretd_x_array);
		Type *soretd_y_array_data = (Type*)PyArray_DATA(soretd_y_array);
		Type *samples_array_data = (Type*)PyArray_DATA(samples_array);
		Type *opt_weights_scale_data = (Type*)PyArray_DATA(opt_weights_scale);

		return loess<Type>(soretd_x_array_data, soretd_x_length, soretd_y_array_data, samples_array_data, samples_length, neighbours, out_data, opt_weights_scale_data);
	}

	bool loess_helper(NPY_TYPES array_type, PyObject *soretd_x_array, const uint32_t soretd_x_length,
					  PyObject *soretd_y_array,
					  PyObject *samples_array, const uint32_t samples_length,
					  const uint32_t neighbours,
					  PyObject *out_array,
					  PyObject *opt_weights_scale)
	{
		bool result = false;
		switch(array_type)
		{
		case NPY_LONG:
			result = loess_type_helper<NPY_LONG>(soretd_x_array, soretd_x_length, soretd_y_array, samples_array, samples_length, neighbours, out_array, opt_weights_scale);
			break;
		case NPY_ULONG:
			result = loess_type_helper<NPY_ULONG>(soretd_x_array, soretd_x_length, soretd_y_array, samples_array, samples_length, neighbours, out_array, opt_weights_scale);
			break;
		case NPY_LONGLONG:
			result = loess_type_helper<NPY_LONGLONG>(soretd_x_array, soretd_x_length, soretd_y_array, samples_array, samples_length, neighbours, out_array, opt_weights_scale);
			break;
		case NPY_ULONGLONG:
			result = loess_type_helper<NPY_ULONGLONG>(soretd_x_array, soretd_x_length, soretd_y_array, samples_array, samples_length, neighbours, out_array, opt_weights_scale);
			break;
		case NPY_FLOAT:
			result = loess_type_helper<NPY_FLOAT>(soretd_x_array, soretd_x_length, soretd_y_array, samples_array, samples_length, neighbours, out_array, opt_weights_scale);
			break;
		case NPY_DOUBLE:
			result = loess_type_helper<NPY_DOUBLE>(soretd_x_array, soretd_x_length, soretd_y_array, samples_array, samples_length, neighbours, out_array, opt_weights_scale);
			break;
		default:
			break;
		}
		return result;
	}

	PyObject * GetArray(PyObject * object, int &in_out_length, NPY_TYPES *out_type = nullptr)
	{
		if(!object)
			return nullptr;

		PyObject *array = PyArray_FROM_OF(object, NPY_IN_ARRAY);
		if(!array)
		{
			PyErr_SetString(PyExc_TypeError, "Couldn't parse the input arrays.");
			return nullptr;
		}

		struct DataHelper
		{
			PyObject *array = nullptr;
			~DataHelper()
			{
				Py_DECREF(array);
			}
		} array_helper{array};

		if(out_type)
		{
			const NPY_TYPES array_type = static_cast<NPY_TYPES>(PyArray_TYPE(array));		
			*out_type = array_type;
		}		

		const int array_dim = (int)PyArray_NDIM(array);
		if(array_dim != 1)
		{
			PyErr_SetString(PyExc_TypeError, "The array must be 1D.");
			return nullptr;
		}

		const int array_length = (int)PyArray_DIM(array, 0);
		if(in_out_length && array_length != in_out_length)
		{
			PyErr_SetString(PyExc_TypeError, "Invalid array length.");
			return nullptr;
		}
		else
			in_out_length = array_length;

		// do not decref if everything ok
		array_helper.array = nullptr;

		return array;
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
	PyObject *soretd_x = nullptr;
	PyObject *soretd_y = nullptr;
	PyObject *samples = nullptr;
	PyObject *weights = nullptr;

	/* Parse positional and keyword arguments */
	static char* keywords[] = {"soretd_x", "soretd_y", "samples", "neighbours", "weights", nullptr};
	if(!PyArg_ParseTuple(args, "OOOI|O", &soretd_x, &soretd_y, &samples, &neighbours, weights))
	{
		return nullptr;
	}

	if(soretd_x == nullptr || soretd_y == nullptr || samples == nullptr || neighbours < 1)
	{
		PyErr_SetString(PyExc_TypeError, "Couldn't parse the input arrays.");
		return nullptr;
	}

	if(PyArray_API == nullptr)
		import_array();
	
	struct DataHelper
	{
		PyObject *soretd_x = nullptr;
		PyObject *soretd_y = nullptr;
		PyObject *samples = nullptr;

		PyObject *weights = nullptr;

		~DataHelper()
		{
			Py_XDECREF(soretd_x);
			Py_XDECREF(soretd_y);
			Py_XDECREF(samples);
		}
	} data_helper{};

	int soretd_x_array_length{0};
	NPY_TYPES soretd_x_array_type{};
	data_helper.soretd_x = GetArray(soretd_x, soretd_x_array_length, &soretd_x_array_type);
	if(!data_helper.soretd_x || !soretd_x_array_length)
	{
		PyErr_SetString(PyExc_TypeError, "Couldn't parse the X array.");
		return nullptr;
	}

	if(!(soretd_x_array_type == NPY_LONG ||
		 soretd_x_array_type == NPY_ULONG ||
		 soretd_x_array_type == NPY_LONGLONG ||
		 soretd_x_array_type == NPY_ULONGLONG ||
		 soretd_x_array_type == NPY_FLOAT ||
		 soretd_x_array_type == NPY_DOUBLE))
	{
		PyErr_SetString(PyExc_TypeError, "Invalid X array type.");
		return nullptr;
	}

	NPY_TYPES soretd_y_array_type{};
	data_helper.soretd_y = GetArray(soretd_y, soretd_x_array_length, &soretd_y_array_type);
	if(!data_helper.soretd_y)
	{
		PyErr_SetString(PyExc_TypeError, "Couldn't parse the Y array.");
		return nullptr;
	}

	int samples_array_length{0};
	NPY_TYPES samples_array_type{};
	data_helper.samples = GetArray(soretd_y, samples_array_length, &samples_array_type);
	if(!data_helper.samples)
	{
		PyErr_SetString(PyExc_TypeError, "Couldn't parse the samples array.");
		return nullptr;
	}

	int weights_array_length{0};
	NPY_TYPES weights_array_type{};
	data_helper.weights = GetArray(weights, neighbours, &weights_array_type);

	/* Build the output array. */
	npy_intp dims{samples_array_length};
	PyObject *out_array = PyArray_SimpleNew(1, &dims, soretd_x_array_type);
	if(out_array == nullptr)
	{
		PyErr_SetString(PyExc_RuntimeError, "Couldn't build output array.");
		return nullptr;
	}		

	if(!loess_helper(soretd_x_array_type,
					 data_helper.soretd_x, soretd_x_array_length, 
					 data_helper.soretd_y, 
					 data_helper.samples, samples_array_length,
					 neighbours, 
					 out_array,
					 data_helper.weights))
	{
		PyErr_SetString(PyExc_RuntimeError, "Failed to calculate loess.");
		Py_DECREF(out_array);
		return nullptr;    /* return nullptr indicates error */
	}

	return out_array;
}

/*
	* List of functions to add to fast_loess in exec_fast_loess().
	*/
static PyMethodDef fast_loess_functions[] = {
	{ "loess", (PyCFunction)fast_loess, METH_VARARGS, fast_loess_doc },
	{ nullptr, nullptr, 0, nullptr } /* marks end of array */
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
	{ 0, nullptr }
};

static PyModuleDef fast_stl_impl_def = {
	PyModuleDef_HEAD_INIT,
	"fast_stl_impl",
	fast_stl_doc,
	0,              /* m_size */
	nullptr,           /* m_methods */
	fast_loess_slots,
	nullptr,           /* m_traverse */
	nullptr,           /* m_clear */
	nullptr,           /* m_free */
};

PyMODINIT_FUNC PyInit_fast_stl_impl()
{
	return PyModuleDef_Init(&fast_stl_impl_def);
}
