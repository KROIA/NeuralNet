#pragma once

#include "NeuralNet_base.h"
#include "QSFML_EditorWidget.h"

namespace NeuralNet
{
	namespace Visualisation
	{
		class Utilities
		{
		public:

			static sf::Color signalColor(float value)
			{
				value = value;
				return QSFML::Color::lerpLinear({ m_lowValueColor, m_mediumValueColor, m_highValueColor }, (value + 1.f) / 2.f);
			}
			static float signalWidth(float weight)
			{
				weight = sqrt(std::abs(weight));
				if (weight < 1)
					return 1;
				return weight;
			}

			static sf::Color m_lowValueColor;
			static sf::Color m_mediumValueColor;
			static sf::Color m_highValueColor;
		};
	}
}
