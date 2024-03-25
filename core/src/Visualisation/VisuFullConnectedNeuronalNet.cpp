#include "Visualisation/VisuFullConnectedNeuronalNet.h"


namespace NeuralNet
{
	namespace Visualisation
	{


		VisuFullConnectedNeuronalNet::VisuFullConnectedNeuronalNet(
			NeuralNet::FullConnectedNeuralNet* net)
			: QSFML::Objects::CanvasObject("VisuFullConnectedNeuronalNet")
		{
			m_net = net;
			m_layerSpacing = 100;
			m_neuronSpacing = 50;
			m_neuronRadius = 10;
			m_connectionWidth = 2;

			m_painter = new Painter(this);
			addComponent(m_painter);
		}
		VisuFullConnectedNeuronalNet::~VisuFullConnectedNeuronalNet()
		{

		}



		void VisuFullConnectedNeuronalNet::Painter::drawComponent(
			sf::RenderTarget& target, 
			sf::RenderStates states) const
		{
			//QSFML_UNUSED(states);
			NeuralNet::FullConnectedNeuralNet* m_net = m_visu->m_net;
			unsigned int inputNeuronCount = m_net->getInputCount();
			unsigned int outputNeuronCount = m_net->getOutputCount();
			unsigned int hiddenLayerCount = m_net->getHiddenLayerCount();
			unsigned int hiddenLayerSize = m_net->getHiddenLayerSize();

			unsigned int layerSpacing = m_visu->m_layerSpacing;
			unsigned int neuronSpacing = m_visu->m_neuronSpacing;
			unsigned int neuronRadius = m_visu->m_neuronRadius;
			unsigned int connectionWidth = m_visu->m_connectionWidth;

			QSFML::Components::LinePainter linePainter;
			QSFML::Components::PointPainter pointPainter;

			


			for (unsigned int i = 0; i < inputNeuronCount; i++)
			{
				float neuronValue = m_net->getInputValue(i);
				pointPainter.addPoint(sf::Vector2f(0, i * neuronSpacing), neuronRadius,
					signalColor(neuronValue));
			}

			for (unsigned int x = 0; x < hiddenLayerCount; x++)
			{
				for (unsigned int y = 0; y < hiddenLayerSize; y++)
				{
					float neuronValue = m_net->getHiddenValue(x, y);
					pointPainter.addPoint(sf::Vector2f(layerSpacing + x * layerSpacing, y * neuronSpacing), neuronRadius,
						signalColor(neuronValue));
				}
			}

			for (unsigned int i = 0; i < outputNeuronCount; i++)
			{
				float neuronValue = m_net->getOutputValue(i);
				pointPainter.addPoint(sf::Vector2f((hiddenLayerCount +1)* layerSpacing, i * neuronSpacing), neuronRadius,
					signalColor(neuronValue));
			}

			// Draw connections
			float xOffset = 0;
			for (unsigned int y = 0; y < hiddenLayerSize; y++)
			{
				for (unsigned int z = 0; z < inputNeuronCount; z++)
				{
					float weight = m_net->getWeight(0, y, z);
					linePainter.addLine(sf::Vector2f(xOffset, z * neuronSpacing),
						sf::Vector2f(xOffset + layerSpacing, y * neuronSpacing),
						signalColor(weight), connectionWidth);
				}
			}


			for (unsigned int x = 1; x < hiddenLayerCount; x++)
			{
				for (unsigned int y = 0; y < hiddenLayerSize; y++)
				{
					for (unsigned int z = 0; z < hiddenLayerSize; z++)
					{
						float weight = m_net->getWeight(x, y, z);
						linePainter.addLine(sf::Vector2f(xOffset + x * layerSpacing, z * neuronSpacing),
														 sf::Vector2f(xOffset + (x + 1) * layerSpacing, y * neuronSpacing),
							signalColor(weight), connectionWidth);
					}
				}
			}

			for (unsigned int y = 0; y < hiddenLayerSize; y++)
			{
				for (unsigned int z = 0; z < outputNeuronCount; z++)
				{
					float weight = m_net->getWeight(hiddenLayerCount, y, z);
					linePainter.addLine(sf::Vector2f(xOffset + hiddenLayerCount * layerSpacing, y * neuronSpacing),
																				 sf::Vector2f(xOffset + (hiddenLayerCount + 1) * layerSpacing, z * neuronSpacing),
						signalColor(weight), connectionWidth);
				}
			}



			linePainter.draw(target, states);
			pointPainter.draw(target, states);
		}
	}
}