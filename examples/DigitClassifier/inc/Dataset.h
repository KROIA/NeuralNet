#pragma once

#include <vector>
#include <string>
#include <SFML/Graphics.hpp>


class Dataset
{
public:
	struct DataPoint
	{
		std::vector<float> features;
		std::vector<float> labels;
	};
	Dataset();
	Dataset(sf::Vector2u dimension);
	~Dataset();

	bool load(const std::string& path);
	const std::vector<DataPoint>& getData() const { return m_data; }

	size_t getInputSize() const { return m_data[0].features.size(); }
	size_t getOutputSize() const { return m_data[0].labels.size(); }
	sf::Vector2u getDimensions() const { return m_dimensions; }

private:
	std::vector<float> loadImage(const std::string& path);
	std::vector<float> centerImage(const std::vector<float> &orig);

	std::vector<DataPoint> m_data;
	sf::Vector2u m_dimensions;
};