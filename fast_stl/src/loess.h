#ifndef c_loess_h__
#define c_loess_h__

#include <stdint.h>
#include <stdio.h>
#include <algorithm>

template <typename T>
struct WindowIterator
{
	const T *x;
	const T *y;
	T *distance;

	bool operator<(const WindowIterator& rht) const
	{
		return x < rht.x;
	}

	bool operator<=(const WindowIterator& rht) const
	{
		return x <= rht.x;
	}

	WindowIterator& operator++()
	{
		++x;
		++y;
		++distance;

		return *this;
	}
};

template <typename T>
WindowIterator<T> operator+(const WindowIterator<T>& value, const uint32_t advance)
{
	return {
		value.x + advance,
		value.y + advance,
		value.distance + advance
	};
}

template <typename T>
WindowIterator<T> operator-(const WindowIterator<T>& value, const uint32_t advance)
{
	return {
		value.x - advance,
		value.y - advance,
		value.distance - advance
	};
}

template <typename T>
void find_neighbours(WindowIterator<T> &window_first, WindowIterator<T> &window_last, const WindowIterator<T> &last, const uint32_t neighbours, const T pivot)
{
	// move window to the left if too close to the last
	if(last - neighbours < window_first)
	{
		window_first = last - neighbours;
	}

	// move window so pivot inside
	while(*window_last.x < pivot)
	{
		++window_first;
		++window_last;
	}

	// fill distances in last window
	for(WindowIterator<T> cursor = window_first; cursor <= window_last; ++cursor)
	{
		T& distance = *cursor.distance;

		bool is_pivot_grater = *cursor.x < pivot;
		if(is_pivot_grater)
			distance = pivot - *cursor.x;
		else
			distance = *cursor.x - pivot;
	}

	T current_window_length = *window_first.distance + *window_last.distance;

	// find smallest distance window
	while(window_last < last)
	{
		const T next_left_distance = pivot - *window_first.x;
		const T next_right_distance = *window_last.x - pivot;

		const T next_window_length = next_left_distance + next_right_distance;

		// move window
		if(next_window_length < current_window_length)
		{
			// advance window
			++window_first;
			++window_last;

			*window_first.distance = next_left_distance;
			*window_last.distance = next_right_distance;

			current_window_length = next_window_length;
		}
		else
			break;
	}
}

template <typename T>
T calculate(const WindowIterator<T> &window_first, const WindowIterator<T> &window_last, T *weights)
{
	T weight_sum{0.0f};

	T *current_weight = weights;
	const T max_distance = std::max(*window_first.distance, *(window_last.distance));
	for(WindowIterator<T> cursor = window_first; cursor <= window_last; ++cursor, ++current_weight)
	{
		const T &distance = *cursor.distance;
		T &weight = *current_weight;

		const T scaled_distance = distance / max_distance;
		weight = std::min(T{1.0}, std::max(T{0.0}, scaled_distance));

		weight = (1 - weight * weight  * weight); // (1 - weight ^ 3)
		weight = weight * weight * weight; // (1 - weight ^ 3) ^ 3
		weight_sum += weight;
	}

	T g{0};
	current_weight = weights;
	for(WindowIterator<T> cursor = window_first; cursor <= window_last; ++cursor, ++current_weight)
	{
		g += (*cursor.y) * (*current_weight);
	}

	g /= weight_sum;

	return g;
}

template <typename T>
bool loess(const T *soretd_x_begin, const uint32_t soretd_x_size, const T *soretd_y_begin, const T *sample_x_begin, const uint32_t num_samples, uint32_t neighbours, T *g)
{
	if(!soretd_x_begin || !soretd_y_begin || neighbours == 0)
		return false;

	if(neighbours > soretd_x_size)
		neighbours = soretd_x_size;
	--neighbours;

	T *distances = new T[soretd_x_size];
	T *weights = new T[neighbours + 1];

	WindowIterator<T> window_first{soretd_x_begin, soretd_y_begin, distances};
	WindowIterator<T> window_last(window_first + neighbours);

	const WindowIterator<T> last{&*(soretd_x_begin + soretd_x_size - 1), nullptr, nullptr};

	for(uint32_t iteration = 0; iteration < num_samples; ++iteration)
	{
		find_neighbours(window_first, window_last, last, neighbours, sample_x_begin[iteration]);

		g[iteration] = calculate(window_first, window_last, weights);
	}

	return true;
}

#endif // c_loess_h__
