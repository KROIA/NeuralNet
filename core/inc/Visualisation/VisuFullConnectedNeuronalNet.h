#pragma once

#include "NeuralNet_base.h"
#include "QSFML_EditorWidget.h"
#include "SimpleImpl/Nets/FullConnectedNeuralNet.h"

namespace NeuralNet
{
	namespace Visualisation
	{
		class NEURAL_NET_EXPORT VisuFullConnectedNeuronalNet : public QObject, public QSFML::Objects::CanvasObject
		{
			Q_OBJECT
			friend class FullConnectedNeuralNet;
			VisuFullConnectedNeuronalNet(NeuralNet::FullConnectedNeuralNet* net);
		public:
			
			~VisuFullConnectedNeuronalNet();

			void update() override;

		private slots:
			void onTimerFired();
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
					value = m_visu->m_signalSatturation * value;
					return QSFML::Color::lerpLinear({ m_lowValueColor, m_mediumValueColor, m_highValueColor }, (value + 1.f) / 2.f);
				}
				float signalWidth(float weight) const
				{
					weight = sqrt(std::abs(weight * m_visu->m_connectionWidth));
					if(weight < 1)
						return 1;
					return weight;
				}
				VisuFullConnectedNeuronalNet* m_visu;
				sf::Color m_lowValueColor = sf::Color::Red;
				sf::Color m_mediumValueColor = sf::Color(150,150,150);
				sf::Color m_highValueColor = sf::Color::Green;
			};


			NeuralNet::FullConnectedNeuralNet* m_net;
			Painter* m_painter;

			unsigned int m_layerSpacing;
			unsigned int m_neuronSpacing;
			unsigned int m_neuronRadius;
			unsigned int m_connectionWidth;
			float m_signalSatturation;

			QTimer m_timer;


		};
	}
}