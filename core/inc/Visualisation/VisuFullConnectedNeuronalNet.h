#pragma once

#include "NeuralNet_base.h"
#include "QSFML_EditorWidget.h"
#include "SimpleImpl/Nets/FullConnectedNeuralNet.h"

namespace NeuralNet
{
	namespace Visualisation
	{
		class NEURAL_NET_EXPORT VisuFullConnectedNeuronalNet : public QSFML::Objects::CanvasObject
		{
		public:
			VisuFullConnectedNeuronalNet(NeuralNet::FullConnectedNeuralNet* net);
			~VisuFullConnectedNeuronalNet();



		private:
			class NEURAL_NET_EXPORT Painter : public QSFML::Components::Drawable
			{
			public:
				Painter(VisuFullConnectedNeuronalNet *visu, const std::string& name = "VisuFullConnectedNeuronalNetPainter")
					: Drawable(name)
					, m_visu(visu)
				{

				}

				void drawComponent(sf::RenderTarget& target, sf::RenderStates states) const override;
				sf::Color signalColor(float value) const
				{
					return QSFML::Color::lerpLinear(m_lowValueColor, m_highValueColor, (value-1.f)/2.f);
				}
				VisuFullConnectedNeuronalNet* m_visu;
				sf::Color m_lowValueColor = sf::Color::Red;
				sf::Color m_highValueColor = sf::Color::Green;
			};


			NeuralNet::FullConnectedNeuralNet* m_net;
			Painter* m_painter;

			unsigned int m_layerSpacing;
			unsigned int m_neuronSpacing;
			unsigned int m_neuronRadius;
			unsigned int m_connectionWidth;


		};
	}
}