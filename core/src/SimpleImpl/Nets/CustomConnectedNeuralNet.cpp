#include "SimpleImpl/Nets/CustomConnectedNeuralNet.h"
#include "SimpleImpl/NetworkComponents/InputNeuron.h"

namespace NeuralNet
{
	CustomConnectedNeuralNet::CustomConnectedNeuralNet(
		unsigned int inputSize,
		unsigned int outputSize)
		: NeuralNetBase(inputSize, outputSize)
	{
		m_outputValues = std::vector<float>(outputSize, 0);
	}

	CustomConnectedNeuralNet::~CustomConnectedNeuralNet()
	{
		std::vector<CustomConnectedNeuralNetPainter*> painters = m_painters;
		m_painters.clear();
		for (auto painter : painters)
		{
			painter->deleteThis();
		}
		destroyNetwork();
	}


	void CustomConnectedNeuralNet::addConnection(const ConnectionInfo& connectionInfo)
	{
		m_networkBuilt = false;
		m_buildingConnections.push_back(connectionInfo);
	}
	void CustomConnectedNeuralNet::addConnection(NeuronID fromNeuronID, NeuronID toNeuronID, float weight)
	{
		m_networkBuilt = false;
		ConnectionInfo info;
		info.fromNeuronID = fromNeuronID;
		info.toNeuronID = toNeuronID;
		info.weight = weight;
		m_buildingConnections.push_back(info);
	}
	void CustomConnectedNeuralNet::setConnections(const std::vector<ConnectionInfo>& connections)
	{
		m_networkBuilt = false;
		m_buildingConnections = connections;
	}

	void CustomConnectedNeuralNet::buildNetwork()
	{
		CustomConnectedNeuralNetBuilder::buildNetwork(m_buildingConnections, getInputCount(), getOutputCount(), m_networkData);
		m_networkBuilt = true;
	}
	void CustomConnectedNeuralNet::destroyNetwork()
	{
		m_networkBuilt = false;
		CustomConnectedNeuralNetBuilder::destroyNetwork(m_networkData);
	}

	void CustomConnectedNeuralNet::setInputValues(const std::vector<float>& values)
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
	void CustomConnectedNeuralNet::setInputValue(unsigned int index, float values)
	{
		if(index < m_inputValues.size())
			m_inputValues[index] = values;
	}
	std::vector<float> CustomConnectedNeuralNet::getOutputValues() const
	{
		return m_outputValues;
	}
	float CustomConnectedNeuralNet::getOutputValue(unsigned int index) const
	{
		if(index < m_outputValues.size())
			return m_outputValues[index];
		return 0.0f;
	
	}

	void CustomConnectedNeuralNet::update()
	{
		if(!m_networkBuilt)	
			return;
		// Set input Values
		std::vector<Layer> &layers = m_networkData.layers;
		if(layers.size() == 0)
			return;

		Layer& inputLayer = layers[0];
		Layer& outputLayer = layers[layers.size()-1];

		if(inputLayer.neurons.size() != m_inputValues.size())
			return;
		for (unsigned int i = 0; i < m_inputValues.size(); ++i)
		{
			InputNeuron* neuron = dynamic_cast<InputNeuron*>(inputLayer.neurons[i]);
			if(neuron)
				neuron->setValue(m_inputValues[i]);
			else
			{
				// Error
			}
		}

		for (auto& layer : layers)
		{
			for (auto& neuron : layer.neurons)
			{
				neuron->update();
			}
		}
		for (unsigned int i = 0; i < outputLayer.neurons.size(); ++i)
		{
			m_outputValues[i] = outputLayer.neurons[i]->getOutput();
		}
	}

	CustomConnectedNeuralNet::CustomConnectedNeuralNetPainter* CustomConnectedNeuralNet::createVisualisation()
	{
		CustomConnectedNeuralNetPainter* painter = new CustomConnectedNeuralNetPainter(this);
		m_painters.push_back(painter);
		return painter;
	}

	void CustomConnectedNeuralNet::removePainter(CustomConnectedNeuralNetPainter* painter)
	{
		auto it = std::find(m_painters.begin(), m_painters.end(), painter);
		if(it != m_painters.end())
			m_painters.erase(it);
	}





}