#include "SimpleImpl/Nets/CustomConnectedNeuralNet.h"
#include "Visualisation/CustomConnectedNeuralNetPainter.h"
#include "Visualisation/Utilities.h"


namespace NeuralNet
{
	namespace Visualisation
	{
		CustomConnectedNeuralNetPainter::CustomConnectedNeuralNetPainter(
			CustomConnectedNeuralNet* net,
			const std::string& name)
			: QSFML::Components::Drawable(name)
			, m_neuralNet(net)
		{

		}

		CustomConnectedNeuralNetPainter::~CustomConnectedNeuralNetPainter()
		{
			m_neuralNet->removePainter(this);
			m_neuralNet = nullptr;
		}
		void CustomConnectedNeuralNetPainter::buildNetwork()
		{
			//unsigned int neuronRadius = 5;

			std::unordered_map<Neuron::ID, NeuronPainterData> cpy = m_neuronPainters;
			/*for (auto& neuronPainter : cpy)
			{
				delete neuronPainter.second.painter;
			}*/
			m_neuronPainters.clear();

			NetworkData& networkData = m_neuralNet->m_networkData;
			for (auto neuronIt : networkData.neurons)
			{
				NeuronPainter* painter = neuronIt.second->createVisualisation();
				NeuronPainterData data;
				data.id = neuronIt.first;
				data.painter = painter;
				data.painter->setCanvasParent(m_canvasParent);
				m_neuronPainters[neuronIt.first] = data;			
			}

			size_t maxLayerSize = 0;
			for (auto& layer : networkData.layers)
			{
				if (layer.neurons.size() > maxLayerSize)
					maxLayerSize = layer.neurons.size();
			}

			float xOffset = 0;
			for (auto& layer : networkData.layers)
			{
				int neuronIndex = 0;
				float yOffset = (maxLayerSize - layer.neurons.size()) * m_neuronSpacing / 2;
				for (auto& neuron : layer.neurons)
				{
					//float signalValue = neuron->getOutput();
					sf::Vector2f position(xOffset, yOffset + neuronIndex * m_neuronSpacing);
					//pointPainter.addPoint(position, neuronRadius,
					//	Utilities::signalColor(signalValue * m_signalSatturation));

					NeuronPainterData &data = m_neuronPainters[neuron->getID()];
					const auto& oldDataIt = cpy.find(neuron->getID());
					if (oldDataIt != cpy.end())
					{
						position = oldDataIt->second.position;
					}
					data.position = position;
					data.painter->setPosition(position);
					data.painter->setRadius(m_neuronRadius);
					//neuronPositions[neuron] = position;

					neuronIndex++;
				}
				xOffset += m_layerSpacing;
			}

		}
		void CustomConnectedNeuralNetPainter::setNeuronRadius(float radius)
		{ 
			m_neuronRadius = radius;
			for (auto& neuronPainter : m_neuronPainters)
			{
				neuronPainter.second.painter->setRadius(radius);
			}
		}

		void CustomConnectedNeuralNetPainter::destroyNetwork()
		{
			m_neuronPainters.clear();
		}

		void CustomConnectedNeuralNetPainter::resetPositions()
		{
			NetworkData& networkData = m_neuralNet->m_networkData;

			size_t maxLayerSize = 0;
			for (auto& layer : networkData.layers)
			{
				if (layer.neurons.size() > maxLayerSize)
					maxLayerSize = layer.neurons.size();
			}

			float xOffset = 0;
			for (auto& layer : networkData.layers)
			{
				int neuronIndex = 0;
				float yOffset = (maxLayerSize - layer.neurons.size()) * m_neuronSpacing / 2;
				for (auto& neuron : layer.neurons)
				{
					//float signalValue = neuron->getOutput();
					sf::Vector2f position(xOffset, yOffset + neuronIndex * m_neuronSpacing);
					//pointPainter.addPoint(position, neuronRadius,
					//	Utilities::signalColor(signalValue * m_signalSatturation));

					NeuronPainterData& data = m_neuronPainters[neuron->getID()];
					data.position = position;
					data.painter->setPosition(position);
					
					//neuronPositions[neuron] = position;

					neuronIndex++;
				}
				xOffset += m_layerSpacing;
			}
		}

		void CustomConnectedNeuralNetPainter::resetLayerPosition(unsigned int layer, const sf::Vector2f& position, const sf::Vector2f& spacing)
		{
			NetworkData& networkData = m_neuralNet->m_networkData;
			if (layer < networkData.layers.size())
			{
				Layer& layerData = networkData.layers[layer];
				for (size_t i=0; i<layerData.neurons.size(); ++i)
				{
					NeuronPainterData& data = m_neuronPainters[layerData.neurons[i]->getID()];
					data.position = spacing * (float)i + position;
					data.painter->setPosition(data.position);
				}
			}
		}

		Neuron::ID CustomConnectedNeuralNetPainter::getNeuronAtPosition(const sf::Vector2f& position, bool& success) const
		{
			for (auto& neuronPainter : m_neuronPainters)
			{
				if (neuronPainter.second.painter->contains(position))
				{
					success = true;
					return neuronPainter.first;
				}
			}
			success = false;
			return 0;
		}
		void CustomConnectedNeuralNetPainter::setNeuronPosition(Neuron::ID id, const sf::Vector2f& position)
		{
			auto it = m_neuronPainters.find(id);
			if (it != m_neuronPainters.end())
			{
				it->second.position = position;
				it->second.painter->setPosition(position);
			}
		}

		void CustomConnectedNeuralNetPainter::moveLayer(unsigned int layer, const sf::Vector2f& offset)
		{
			NetworkData& networkData = m_neuralNet->m_networkData;
			if(layer < networkData.layers.size())
			{
				Layer& layerData = networkData.layers[layer];
				for (auto& neuron : layerData.neurons)
				{
					NeuronPainterData& data = m_neuronPainters[neuron->getID()];
					data.position += offset;
					data.painter->setPosition(data.position);
				}
			}
		}
		void CustomConnectedNeuralNetPainter::moveNeuron(Neuron::ID id, const sf::Vector2f& offset)
		{
			auto it = m_neuronPainters.find(id);
			if (it != m_neuronPainters.end())
			{
				it->second.position += offset;
				it->second.painter->move(offset);
			}
		}


		void CustomConnectedNeuralNetPainter::enableNeuronGraph(Neuron::ID id, bool enable)
		{
			auto it = m_neuronPainters.find(id);
			if (it != m_neuronPainters.end())
			{
				it->second.painter->enableGraph(enable);
			}
		}
		void CustomConnectedNeuralNetPainter::enableNeuronGraph(bool enable)
		{
			for (auto& pair : m_neuronPainters)
			{
				pair.second.painter->enableGraph(enable);
			}
		}
		void CustomConnectedNeuralNetPainter::enableNeuronGraphOfLayer(unsigned int layer, bool enable)
		{
			NetworkData& networkData = m_neuralNet->m_networkData;
			if (layer < networkData.layers.size())
			{
				Layer& layerData = networkData.layers[layer];
				for (auto& neuron : layerData.neurons)
				{
					NeuronPainterData& data = m_neuronPainters[neuron->getID()];
					data.painter->enableGraph(enable);
				}
			}
		}

		void CustomConnectedNeuralNetPainter::enableNeuronText(Neuron::ID id, bool enable)
		{
			auto it = m_neuronPainters.find(id);
			if (it != m_neuronPainters.end())
			{
				it->second.painter->enableText(enable);
			}
		}
		void CustomConnectedNeuralNetPainter::enableNeuronText(bool enable)
		{
			for (auto& pair : m_neuronPainters)
			{
				pair.second.painter->enableText(enable);
			}
		}
		void CustomConnectedNeuralNetPainter::enableNeuronTextOfLayer(unsigned int layer, bool enable)
		{
			NetworkData& networkData = m_neuralNet->m_networkData;
			if (layer < networkData.layers.size())
			{
				Layer& layerData = networkData.layers[layer];
				for (auto& neuron : layerData.neurons)
				{
					NeuronPainterData& data = m_neuronPainters[neuron->getID()];
					data.painter->enableText(enable);
				}
			}
		}


		void CustomConnectedNeuralNetPainter::drawComponent(
			sf::RenderTarget& target,
			sf::RenderStates states) const
		{
			NetworkData& networkData = m_neuralNet->m_networkData;

			

			QSFML::Components::LinePainter linePainter;
			QSFML::Components::PointPainter pointPainter;
			pointPainter.setVerteciesCount(20);


			//std::unordered_map<Neuron*, sf::Vector2f> neuronPositions;
			//neuronPositions.reserve(networkData.neurons.size());

			/*size_t maxLayerSize = 0;
			for (auto& layer : networkData.layers)
			{
				if (layer.neurons.size() > maxLayerSize)
					maxLayerSize = layer.neurons.size();
			}

			float xOffset = 0;
			for (auto& layer : networkData.layers)
			{
				int neuronIndex = 0;
				float yOffset = (maxLayerSize - layer.neurons.size()) * neuronSpacing / 2;
				for (auto& neuron : layer.neurons)
				{
					float signalValue = neuron->getOutput();
					sf::Vector2f position(xOffset, yOffset + neuronIndex * neuronSpacing);
					pointPainter.addPoint(position, neuronRadius,
						Utilities::signalColor(signalValue * m_signalSatturation));

					neuronPositions[neuron] = position;

					neuronIndex++;
				}
				xOffset += layerSpacing;
			}*/

			

			for (auto& connection : networkData.connections)
			{
				float weight = connection->getWeight();
				float signalValue = connection->getOutputValue();
				float signalW = Utilities::signalWidth(weight * m_connectionWidth);
				
				sf::Vector2f start;
				sf::Vector2f end;

				const auto& startIt = m_neuronPainters.find(connection->getStartNeuron()->getID());
				const auto& endIt = m_neuronPainters.find(connection->getEndNeuron()->getID());

				if (startIt != m_neuronPainters.end() && endIt != m_neuronPainters.end())
				{
					start = startIt->second.painter->getOutputConnectionPoint();
					end = endIt->second.painter->getInputConnectionPoint();
				}
				else
				{
					continue;
				}

				linePainter.addLine(
					start,
					end,
					Utilities::signalColor(signalValue * m_signalSatturation), signalW);
			}



			linePainter.draw(target, states);
			//pointPainter.draw(target, states);

			for (auto& neuronPainter : m_neuronPainters)
			{
				neuronPainter.second.painter->draw(target, states);
				//neuronPositions[neuronPainter.second.id] = neuronPainter.second.position;
			}
		}
	}
}