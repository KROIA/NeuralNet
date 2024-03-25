#include "SimpleImpl/Nets/FullConnectedNeuralNet.h"

namespace NeuralNet
{
	FullConnectedNeuralNet::FullConnectedNeuralNet(
		unsigned int inputSize,
		unsigned int outputSize,
		unsigned int hiddenLayerCount,
		unsigned int hiddenLayerSize)
		: NeuralNetBase(inputSize, outputSize)
		, m_hiddenLayerCount(hiddenLayerCount)
		, m_hiddenLayerSize(hiddenLayerSize)
		, m_inputValues(inputSize, 0)
		, m_outputValues(outputSize, 0)
	{
		buildNetwork();
	}

	FullConnectedNeuralNet::~FullConnectedNeuralNet()
	{
		destroyNetwork();
	}



	void FullConnectedNeuralNet::setInputValues(const std::vector<float>& values)
	{
		if (values.size() != getInputCount())
		{
			size_t minSize = std::min(values.size(), (size_t)getInputCount());
			std::copy(values.begin(), values.begin() + minSize, m_inputValues.begin());
		}
		else
		{
			m_inputValues = values;
		}
	}
	void FullConnectedNeuralNet::setInputValue(unsigned int index, float value)
	{
		if(index < getInputCount())
			m_inputValues[index] = value;
	}
	std::vector<float> FullConnectedNeuralNet::getOutputValues() const
	{
		return m_outputValues;
	}
	float FullConnectedNeuralNet::getOutputValue(unsigned int index) const
	{
		if(index < getOutputCount())
			return m_outputValues[index];
		return 0.0f;
	}

	void FullConnectedNeuralNet::update()
	{
		if (m_hiddenLayerCount == 0)
			return;
		// Set input Values
		LayerData &inputLayer = m_layers[0];
		LayerData &outputLayer = m_layers[m_hiddenLayerCount]; // Index + 1 because of the input layer
		for (unsigned int i = 0; i < inputLayer.neurons.size(); ++i)
		{
			inputLayer.neurons[i]->addInputValue(m_inputValues[i]);
		}
		for (auto& layer : m_layers)
		{
			for (auto& neuron : layer.neurons)
			{
				neuron->clearValue();
			}
		}
		for(auto &layer : m_layers)
		{
			for (auto& connection : layer.inputConnections)
			{
				connection->passValue();
			}
			for(auto &neuron : layer.neurons)
			{
				neuron->update();

			}
		}
		for (unsigned int i = 0; i < outputLayer.neurons.size(); ++i)
		{
			m_outputValues[i] = outputLayer.neurons[i]->getOutput();
		}
	}

	std::vector<float> FullConnectedNeuralNet::getWeights() const
	{
		std::vector<float> weights( m_hiddenLayerCount * m_hiddenLayerSize * m_hiddenLayerSize +
									m_hiddenLayerSize * getInputCount(), 0);

		size_t index = 0;
		for (auto& layer : m_layers)
		{
			for (auto& connection : layer.inputConnections)
			{
				weights[index++] = connection->getWeight();
			}
		}

		return weights;
	}
	float FullConnectedNeuralNet::getWeight(unsigned int layerIdx, unsigned int neuronIdx, unsigned int inputIdx) const
	{
		layerIdx++; // Skip the input layer
		if (layerIdx > m_hiddenLayerCount+1)
			return 0.0f;
		if (neuronIdx >= m_hiddenLayerSize)
			return 0.0f;
		if (inputIdx >= getInputCount())
			return 0.0f;
		unsigned int offset = 0;
		if (layerIdx == 1)
		{
			offset = neuronIdx * getInputCount();
		}
		else if(layerIdx < m_hiddenLayerCount+1)
		{
			offset = neuronIdx * m_hiddenLayerSize;
		}
		else
		{
			offset = neuronIdx * getOutputCount();
		}
		return m_layers[layerIdx].inputConnections[offset + inputIdx]->getWeight();
	}
	void FullConnectedNeuralNet::setWeights(const std::vector<float>& weights)
	{
		if (weights.size() != m_hiddenLayerCount * m_hiddenLayerSize * m_hiddenLayerSize +
			m_hiddenLayerSize * getInputCount())
		{
			return;
		}
		size_t index = 0;
		for (auto& layer : m_layers)
		{
			for (auto& connection : layer.inputConnections)
			{
				connection->setWeight(weights[index++]);
			}
		}
	}
	void FullConnectedNeuralNet::setWeight(unsigned int layerIdx, unsigned int neuronIdx, unsigned int inputIdx, float weight)
	{
		layerIdx++; // Skip the input layer
		if (layerIdx > m_hiddenLayerCount)
			return;
		if (neuronIdx >= m_hiddenLayerSize)
			return;
		if (inputIdx >= getInputCount())
			return;
		unsigned int offset = 0;
		if (layerIdx == 1)
		{
			offset = neuronIdx * getInputCount();
		}
		else
		{
			offset = m_hiddenLayerSize * getInputCount() + 
				(layerIdx - 2) * m_hiddenLayerSize * m_hiddenLayerSize + 
				neuronIdx * m_hiddenLayerSize;
		}
		m_layers[layerIdx].inputConnections[offset + inputIdx]->setWeight(weight);
	}

	Activation::Type FullConnectedNeuralNet::getActivationType(unsigned int layerIdx, unsigned int neuronIdx) const
	{
		layerIdx++; // Skip the input layer
		if (layerIdx >= m_layers.size())
			return Activation::Type::linear;
		if (neuronIdx >= m_layers[layerIdx].neurons.size())
			return Activation::Type::linear;
		return m_layers[layerIdx].neurons[neuronIdx]->getActivationType();
	}
	void FullConnectedNeuralNet::setActivationType(unsigned int layerIdx, unsigned int neuronIdx, Activation::Type type) const
	{
		layerIdx++; // Skip the input layer
		if (layerIdx >= m_layers.size())
			return;
		if (neuronIdx >= m_layers[layerIdx].neurons.size())
			return;
		m_layers[layerIdx].neurons[neuronIdx]->setActivationType(type);	
	}
	void FullConnectedNeuralNet::setActivationType(unsigned int layerIdx, Activation::Type type) const
	{
		layerIdx++; // Skip the input layer
		if (layerIdx >= m_layers.size())
			return;
		for (auto& neuron : m_layers[layerIdx].neurons)
		{
			neuron->setActivationType(type);
		}
	}
	void FullConnectedNeuralNet::setActivationType(Activation::Type type) const
	{
		for (auto& layer : m_layers)
		{
			for (auto& neuron : layer.neurons)
			{
				neuron->setActivationType(type);
			}
		}
	}
	float FullConnectedNeuralNet::getInputValue(unsigned int index) const
	{
		if(index < getInputCount())
			return m_inputValues[index];
		return 0.0f;
	}
	float FullConnectedNeuralNet::getHiddenValue(unsigned int layerIdx, unsigned int neuronIdx) const 
	{
		layerIdx++; // Skip the input layer
		if (layerIdx >= m_layers.size())
			return 0.0f;
		if (neuronIdx >= m_layers[layerIdx].neurons.size())
			return 0.0f;
		return m_layers[layerIdx].neurons[neuronIdx]->getOutput();
	}

	void FullConnectedNeuralNet::buildNetwork()
	{
		destroyNetwork();
		m_layers.resize(m_hiddenLayerCount+2);
		// Create the input layer
		LayerData& inputLayer = m_layers[0];
		inputLayer.neurons.resize(getInputCount());
		for (unsigned int i = 0; i < getInputCount(); ++i)
		{
			inputLayer.neurons[i] = new NeuronBase();
		}
		for (unsigned int i = 1; i < m_hiddenLayerCount+1; ++i)
		{
			LayerData& layer = m_layers[i];
			layer.neurons.resize(m_hiddenLayerSize);
			for (unsigned int j = 0; j < m_hiddenLayerSize; ++j)
			{
				layer.neurons[j] = new Neuron();
			}
		}
		
		// Connect the layers
		if (m_hiddenLayerCount >= 1)
		{
			LayerData &firstHiddenLayer = m_layers[1];
			for (unsigned int l = 0; l < m_hiddenLayerSize; ++l)
			{
				for (unsigned int i = 0; i < getInputCount(); ++i)
				{
					Connection* connection = new Connection();
					connection->setWeight(1.0f);
					connection->setStartNeuron(inputLayer.neurons[i]);
					connection->setEndNeuron(firstHiddenLayer.neurons[l]);
					firstHiddenLayer.inputConnections.push_back(connection);
				}
			}
		}
		for (unsigned int i = 2; i < m_hiddenLayerCount+1; ++i)
		{
			LayerData& sendingLayer = m_layers[i-1];
			LayerData& receivingLayer = m_layers[i];
			
			for (auto& receivingNeuron : receivingLayer.neurons)
			{
				for (auto& sendingNeuron : sendingLayer.neurons)
				{
					Connection* connection = new Connection();
					connection->setWeight(1.0f);
					connection->setStartNeuron(sendingNeuron);
					connection->setEndNeuron(receivingNeuron);
					receivingLayer.inputConnections.push_back(connection);
				}
			}
		}

		LayerData& outputLayer = m_layers[m_hiddenLayerCount+1];
		outputLayer.neurons.resize(getOutputCount());
		for (unsigned int i = 0; i < getOutputCount(); ++i)
		{
			outputLayer.neurons[i] = new Neuron();
		}
		for (auto& receivingNeuron : outputLayer.neurons)
		{
			for (auto& sendingNeuron : m_layers[m_hiddenLayerCount].neurons)
			{
				Connection* connection = new Connection();
				connection->setWeight(1.0f);
				connection->setStartNeuron(sendingNeuron);
				connection->setEndNeuron(receivingNeuron);
				outputLayer.inputConnections.push_back(connection);
			}
		}
	}
	void FullConnectedNeuralNet::destroyNetwork()
	{
		for(auto &layer : m_layers)
		{
			for(auto &connection : layer.inputConnections)
				delete connection;
			for(auto &neuron : layer.neurons)
				delete neuron;
		}
		m_layers.clear();
	}
}