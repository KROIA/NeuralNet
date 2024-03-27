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
				VisuFullConnectedNeuronalNet* m_visu;
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