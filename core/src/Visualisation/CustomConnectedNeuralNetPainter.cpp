#include "SimpleImpl/Nets/CustomConnectedNeuralNet.h"


namespace NeuralNet
{
	CustomConnectedNeuralNet::CustomConnectedNeuralNetPainter::CustomConnectedNeuralNetPainter(
		CustomConnectedNeuralNet* net,
		const std::string& name)
		: QSFML::Components::Drawable(name)
		, m_neuralNet(net)
	{

	}

	CustomConnectedNeuralNet::CustomConnectedNeuralNetPainter::~CustomConnectedNeuralNetPainter()
	{
		m_neuralNet->removePainter(this);
		m_neuralNet = nullptr;
	}

	void CustomConnectedNeuralNet::CustomConnectedNeuralNetPainter::drawComponent(
		sf::RenderTarget& target,
		sf::RenderStates states) const
	{
		NetworkData &networkData = m_neuralNet->m_networkData;

		unsigned int layerSpacing = 50;
		unsigned int neuronSpacing = 30;
		unsigned int neuronRadius = 5;

		QSFML::Components::LinePainter linePainter;
		QSFML::Components::PointPainter pointPainter;
		pointPainter.setVerteciesCount(20);

		
		std::unordered_map<Neuron*, sf::Vector2f> neuronPositions;
		neuronPositions.reserve(networkData.neurons.size());



		float xOffset = 0;
		for (auto& layer : networkData.layers)
		{
			int neuronIndex = 0;
			for (auto& neuron : layer.neurons)
			{
				float signalValue = neuron->getOutput();
				pointPainter.addPoint(sf::Vector2f(xOffset, neuronIndex * neuronSpacing), neuronRadius,
					signalColor(signalValue));
				
				neuronPositions[neuron] = sf::Vector2f(xOffset, neuronIndex * neuronSpacing);

				neuronIndex++;
			}

			/*for (auto& connection : layer.inputConnections)
			{
				float weight = connection->getWeight();
				float signalValue = connection->getOutputValue();
				float signalW = signalWidth(weight);
				linePainter.addLine(
					sf::Vector2f(connection->getInput()->getOutputPosition().x, connection->getInput()->getOutputPosition().y),
					sf::Vector2f(connection->getOutput()->getOutputPosition().x, connection->getOutput()->getOutputPosition().y),
					signalColor(signalValue), signalW);
			}*/

			xOffset += layerSpacing;
		}

		for (auto& connection : networkData.connections)
		{
			float weight = connection->getWeight();
			float signalValue = connection->getOutputValue();
			float signalW = signalWidth(weight);
			linePainter.addLine(
				neuronPositions[connection->getStartNeuron()],
				neuronPositions[connection->getEndNeuron()],
				signalColor(signalValue), signalW);
		}



		linePainter.draw(target, states);
		pointPainter.draw(target, states);
	}
}