#include "SimpleImpl/Nets/FullConnectedNet.h"

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
		, m_inputSignals(inputSize, 0)
		, m_outputSignals(outputSize, 0)
	{
		buildNetwork();
	}

	FullConnectedNeuralNet::~FullConnectedNeuralNet()
	{
		destroyNetwork();
	}



	void FullConnectedNeuralNet::setInputValues(const std::vector<float>& signals)
	{
		if (signals.size() != getInputCount())
		{
			size_t minSize = std::min(signals.size(), (size_t)getInputCount());
			std::copy(signals.begin(), signals.begin() + minSize, m_inputSignals.begin());
		}
		else
		{
			m_inputSignals = signals;
		}
	}
	void FullConnectedNeuralNet::setInputValue(unsigned int index, float signal)
	{
		if(index < getInputCount())
			m_inputSignals[index] = signal;
	}
	std::vector<float> FullConnectedNeuralNet::getOutputValues() const
	{
		return m_outputSignals;
	}
	float FullConnectedNeuralNet::getOutputValue(unsigned int index) const
	{
		if(index < getOutputCount())
			return m_outputSignals[index];
		return 0.0f;
	}

	void FullConnectedNeuralNet::update()
	{
		if (m_hiddenLayerCount == 0)
			return;
		// Set input signals
		LayerData &inputLayer = m_layers[0];
		LayerData &outputLayer = m_layers[m_hiddenLayerCount]; // Index + 1 because of the input layer
		for (unsigned int i = 0; i < inputLayer.inputConnections.size(); ++i)
		{
			inputLayer.neurons[i]->addInputValue(m_inputSignals[i]);
		}
		for(auto &layer : m_layers)
		{
			for (auto& connection : layer.inputConnections)
			{
				connection->passSignal();
			}
			for(auto &neuron : layer.neurons)
			{
				neuron->update();
			}
		}
		for (unsigned int i = 0; i < outputLayer.neurons.size(); ++i)
		{
			m_outputSignals[i] = outputLayer.neurons[i]->getOutput();
		}
	}

	void FullConnectedNeuralNet::buildNetwork()
	{
		destroyNetwork();
		m_layers.resize(m_hiddenLayerCount+1);
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
		for (unsigned int i = 1; i < m_hiddenLayerCount; ++i)
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