#ifndef c_loess_h__
#define c_loess_h__

#include <stdio.h>
#include <tchar.h>
#include <vector>
#include <algorithm>
#include <iostream>

struct WindowIterator
{
	const float *x;
	const float *y;
	float *distance;

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

WindowIterator operator+(const WindowIterator& value, const uint32_t advance)
{
	return {
		value.x + advance,
		value.y + advance,
		value.distance + advance
	};
}

WindowIterator operator-(const WindowIterator& value, const uint32_t advance)
{
	return {
		value.x - advance,
		value.y - advance,
		value.distance - advance
	};
}

void find_neighbours(WindowIterator &window_first, WindowIterator &window_last, const WindowIterator &last, const uint32_t neighbours, const float pivot)
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
	for(WindowIterator cursor = window_first; cursor <= window_last; ++cursor)
	{
		float& distance = *cursor.distance;

		bool is_pivot_grater = *cursor.x < pivot;
		if(is_pivot_grater)
			distance = pivot - *cursor.x;
		else
			distance = *cursor.x - pivot;
	}

	float current_window_length = *window_first.distance + *window_last.distance;

	// find smallest distance window
	while(window_last < last)
	{
		const float next_left_distance = pivot - *window_first.x;
		const float next_right_distance = *window_last.x - pivot;

		const float next_window_length = next_left_distance + next_right_distance;

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

float calculate(const WindowIterator &window_first, const WindowIterator &window_last, float *weights)
{
	float weight_sum{0.0f};

	float *current_weight = weights;
	const float max_distance = std::max(*window_first.distance, *(window_last.distance));
	for(WindowIterator cursor = window_first; cursor <= window_last; ++cursor, ++current_weight)
	{
		const float &distance = *cursor.distance;
		float &weight = *current_weight;

		const float scaled_distance = distance / max_distance;
		weight = std::min(1.0f, std::max(0.0f, scaled_distance));

		weight = (1 - weight * weight  * weight); // (1 - weight ^ 3)
		weight = weight * weight * weight; // (1 - weight ^ 3) ^ 3
		weight_sum += weight;
	}

	float g{0};
	current_weight = weights;
	for(WindowIterator cursor = window_first; cursor <= window_last; ++cursor, ++current_weight)
	{
		g += (*cursor.y) * (*current_weight);
	}

	g /= weight_sum;

	return g;
}

std::vector<float> loess(const std::vector<float> &soretd_x, const std::vector<float> &soretd_y, const std::vector<float> &sample_x, uint32_t neighbours)
{
	if(soretd_x.size() != soretd_y.size() || soretd_x.size() < neighbours || neighbours == 0)
		return {};

	if(neighbours > soretd_x.size())
		neighbours = soretd_x.size();
	--neighbours;

	std::vector<float> distances(soretd_x.size(), 0.0f);
	WindowIterator window_first{soretd_x.data(), soretd_y.data(), distances.data()};
	WindowIterator window_last(window_first + neighbours);

	const WindowIterator last{&*(soretd_x.cend() - 1), nullptr, nullptr};

	std::vector<float> weights(neighbours + 1, 0.0f);
	std::vector<float> g(sample_x.size(), 0.0f);

	auto g_it = g.begin();
	for(const auto& sample : sample_x)
	{
		find_neighbours(window_first, window_last, last, neighbours, sample);

		*g_it = calculate(window_first, window_last, weights.data());
		++g_it;
	}

	return g;
}

#endif // c_loess_h__
