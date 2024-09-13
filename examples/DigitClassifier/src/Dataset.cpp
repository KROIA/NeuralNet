#include "Dataset.h"

#include <QDir>
#include <iostream>
#include <thread>
#include <algorithm>
#include <random>

Dataset::Dataset()
{
	m_dimensions = sf::Vector2u(28,28);
}
Dataset::Dataset(sf::Vector2u dimension)
{
	m_dimensions = dimension;
}
Dataset::~Dataset()
{

}

bool Dataset::load(const std::string& path)
{
	// Using Dataset from:
	// https://www.kaggle.com/datasets/jcprogjava/handwritten-digits-dataset-not-in-mnist?resource=download
	// Dataset stored in the folder "dataset" in the root of the project
	// The folder contains 10 subfolders, each one with the images of the digits from 0 to 9

    m_data.clear();

	// create a thread for each folder


	std::vector<std::vector<DataPoint>> data(10);
	std::vector<std::thread*> threads(10, nullptr);
    for(unsigned int i = 0; i < 10; i++)
	{
		// Threads
		std::vector<DataPoint> &d = data[i];
		int label = i;
		std::string folder = path + "/" + std::to_string(i);

		std::thread* thread = new std::thread([this, &d, label, folder]()
			{
			QDir dir(QString::fromStdString(folder));
			QStringList images = dir.entryList(QStringList() << "*.png" << "*.jpg" << "*.jpeg", QDir::Files);
			std::cout << "Loading images from folder " << folder << "..." << std::endl;
			float percentage = 0;
			d.reserve(images.size());
			for (int j = 0; j < images.size(); j += 1)
			{
				std::string image = folder + "/" + images[j].toStdString();
				std::vector<float> data = loadImage(image);
				if (data.size() > 0)
				{
					DataPoint dataPoint;
					dataPoint.features = data;
					dataPoint.labels = std::vector<float>(10, -1.0f);
					dataPoint.labels[label] = 1.0f;
					d.push_back(dataPoint);
				}

				
				if (j % 1000 == 0)
				{
					percentage = (float)(j + 1) / (float)images.size() * 100.0f;
					std::cout << label << ": Loading images from folder " << folder << "..." << percentage << "%" << std::endl;
				}
			}
			std::cout << label << ": Loading images from folder " << folder << "..." << 100 << "%" << std::endl;
			});
		threads[i] = thread;
 
	}


	for (size_t i = 0; i < threads.size(); ++i)
	{
		if (threads[i] != nullptr)
		{
			threads[i]->join();
			delete threads[i];
			m_data.insert(m_data.end(), data[i].begin(), data[i].end());
		}
	}

	// shuffle data
	std::shuffle(m_data.begin(), m_data.end(), std::mt19937(std::random_device()()));
	return true;
}

std::vector<float> Dataset::loadImage(const std::string& path)
{
    std::vector<float> image(m_dimensions.x * m_dimensions.y, 0);

    // Load image from path
    sf::Image inputImage;
    if (!inputImage.loadFromFile(path)) {
        std::cerr << "Error: Unable to load image from " << path << std::endl;
        return image; // Return empty vector on failure
    }

	// Resize image to 28x28
	sf::Vector2u orgSize = inputImage.getSize();
	for (unsigned int x = 0; x < m_dimensions.x; x++)
	{
		for (unsigned int y = 0; y < m_dimensions.y; y++)
		{
			sf::Color color = inputImage.getPixel(x * orgSize.x / m_dimensions.x, y * orgSize.y / m_dimensions.y);
			int gray = ((int)color.r + (int)color.g + (int)color.b + (int)color.a);
			image[x * m_dimensions.y + y] = ((float)gray / 255.0f);
		}
	}
    //return image;
    return centerImage(image);
}

std::vector<float> Dataset::centerImage(const std::vector<float> &orig)
{
	sf::Vector2f massCenterPos(0, 0);
	sf::Vector2i center(m_dimensions.x / 2, m_dimensions.y / 2);
	float mass = 0;
	std::vector<float> centered(m_dimensions.x * m_dimensions.y, 0);
	/*orig[0] = 1;
	orig[m_dimensions.y-1] = 1;
	orig[(m_dimensions.x-1)* m_dimensions.y] = 1;
	orig[(m_dimensions.x)* m_dimensions.y-1] = 1;*/


	for (unsigned int x = 0; x < m_dimensions.x; x++)
	{
		for (unsigned int y = 0; y < m_dimensions.y; y++)
		{
			float subMass = (float)orig[x * m_dimensions.y + y];
			massCenterPos += sf::Vector2f(x, y) * subMass;
			mass += subMass;
		}
	}
	massCenterPos /= mass;
	sf::Vector2i offCenterPos(sf::Vector2i(massCenterPos) - center);
	for (unsigned int x = 0; x < m_dimensions.x; x++)
	{
		for (unsigned int y = 0; y < m_dimensions.y; y++)
		{
			float val = 0;
			sf::Vector2u sourcePos(x + offCenterPos.x, y + offCenterPos.y);
			if (sourcePos.x < m_dimensions.x && sourcePos.y < m_dimensions.y)
			{
				val = orig[sourcePos.x * m_dimensions.y + sourcePos.y];
			}
			centered[x * m_dimensions.y + y] = val;
		}
	}

	return centered;
}