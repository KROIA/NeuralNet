#include "SimpleImpl/Nets/CustomConnectedNeuralNet.h"
#include "SimpleImpl/NetworkComponents/InputNeuron.h"
#include "Visualisation/CustomConnectedNeuralNetPainter.h"

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
		std::vector<Visualisation::CustomConnectedNeuralNetPainter*> painters = m_painters;
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
	void CustomConnectedNeuralNet::addConnection(Neuron::ID fromNeuronID, Neuron::ID toNeuronID)
	{
		addConnection(fromNeuronID, toNeuronID, QSFML::Utilities::RandomEngine::getFloat(-1, 1));
	}
	void CustomConnectedNeuralNet::addConnection(Neuron::ID fromNeuronID, Neuron::ID toNeuronID, float weight)
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
	void CustomConnectedNeuralNet::removeConnection(Neuron::ID fromNeuronID, Neuron::ID toNeuronID)
	{
		for (auto it = m_buildingConnections.begin(); it != m_buildingConnections.end(); ++it)
		{
			if (it->fromNeuronID == fromNeuronID && it->toNeuronID == toNeuronID)
			{
				m_buildingConnections.erase(it);
				break;
			}
		}
	}
	std::vector<ConnectionInfo> CustomConnectedNeuralNet::getConnections() const
	{
		std::vector<ConnectionInfo> connections;
		connections.reserve(m_networkData.connections.size());
		for (auto& connection : m_networkData.connections)
		{
			ConnectionInfo info;
			info.fromNeuronID = connection->getStartNeuron()->getID();
			info.toNeuronID = connection->getEndNeuron()->getID();
			info.weight = connection->getWeight();
			connections.push_back(info);
		}
		return connections;
	}

	void CustomConnectedNeuralNet::buildNetwork()
	{
		
		CustomConnectedNeuralNetBuilder::buildNetwork(
			m_buildingConnections, 
			m_activationFunctions, 
			m_defaultLayerActivationTypes,
			m_defaultActivationType,
			getInputCount(), 
			getOutputCount(),
			m_networkData);
		m_networkBuilt = true;
		for (auto painter : m_painters)
		{
			painter->buildNetwork();
		}
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

	Neuron* CustomConnectedNeuralNet::getNeuron(Neuron::ID id)
	{
		auto it = m_networkData.neurons.find(id);
		if (it != m_networkData.neurons.end())
		{
			return it->second;
		}
		return nullptr;
	}

	Activation::Type CustomConnectedNeuralNet::getActivationType(Neuron::ID id) const
	{
		const auto it = m_networkData.neurons.find(id);
		if (it != m_networkData.neurons.end())
		{
			return it->second->getActivationType();
		}
		return Activation::Type::linear;
	}
	void CustomConnectedNeuralNet::setActivationType(Neuron::ID id, Activation::Type type)
	{
		auto it = m_networkData.neurons.find(id);
		if (it != m_networkData.neurons.end())
		{
			it->second->setActivationType(type);
		}
		m_activationFunctions[id] = type;
	}
	void CustomConnectedNeuralNet::setLayerActivationType(unsigned int layerIdx, Activation::Type type)
	{
		m_defaultLayerActivationTypes[layerIdx] = type;
		if (layerIdx < m_networkData.layers.size())
		{
			for (auto& neuron : m_networkData.layers[layerIdx].neurons)
			{
				neuron->setActivationType(type);
				m_activationFunctions[neuron->getID()] = type;
			}
		}
	}
	void CustomConnectedNeuralNet::setActivationType(Activation::Type type)
	{
		m_defaultActivationType = type;
		for (auto& layer : m_networkData.layers)
		{
			for (auto& neuron : layer.neurons)
			{
				neuron->setActivationType(type);
			}
		}
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

	Visualisation::CustomConnectedNeuralNetPainter* CustomConnectedNeuralNet::createVisualisation()
	{
		Visualisation::CustomConnectedNeuralNetPainter* painter = new Visualisation::CustomConnectedNeuralNetPainter(this);
		m_painters.push_back(painter);
		painter->buildNetwork();
		return painter;
	}

	std::vector<float> CustomConnectedNeuralNet::getWeights() const
	{
		size_t weightCount = getWeightCount();
		std::vector<float> weights(weightCount, 0);

		size_t index = 0;
		for (auto& layer : m_networkData.layers)
		{
			for (auto& neuron : layer.neurons)
			{
				for(auto& connection : neuron->getInputConnections())
					weights[index++] = connection->getWeight();
			}
		}

		return weights;
	}
	float CustomConnectedNeuralNet::getWeight(unsigned int layerIdx, unsigned int neuronIdx, unsigned int inputIdx) const
	{
		if(m_networkData.layers.size() <= layerIdx)
			return 0.0f;
		const Layer& layer = m_networkData.layers[layerIdx];
		if(layer.neurons.size() <= neuronIdx)
			return 0.0f;
		const auto &conn = layer.neurons[neuronIdx]->getInputConnections();
		if(conn.size() <= inputIdx)
			return 0.0f;
		return conn[inputIdx]->getWeight();
	}
	void CustomConnectedNeuralNet::setWeights(const std::vector<float>& weights)
	{
		if (weights.size() != getWeightCount())
			return;
		size_t index = 0;
		for (auto& layer : m_networkData.layers)
		{
			for (auto& neuron : layer.neurons)
			{
				for(auto& connection : neuron->getInputConnections())
					connection->setWeight(weights[index++]);
			}
		}
	}
	void CustomConnectedNeuralNet::setWeight(unsigned int layerIdx, unsigned int neuronIdx, unsigned int inputIdx, float weight)
	{
		if (m_networkData.layers.size() <= layerIdx)
			return;
		const Layer& layer = m_networkData.layers[layerIdx];
		if (layer.neurons.size() <= neuronIdx)
			return;
		const auto& conn = layer.neurons[neuronIdx]->getInputConnections();
		if (conn.size() <= inputIdx)
			return;
		conn[inputIdx]->setWeight(weight);
	}


	void CustomConnectedNeuralNet::learn(const std::vector<float>& expectedOutput)
	{
		if (expectedOutput.size() != getOutputCount())
			return;
		m_backProp.learn(m_networkData.layers, expectedOutput);
	}
	std::vector<float> CustomConnectedNeuralNet::getOutputError(const std::vector<float>& expectedOutput) const
	{
		if (expectedOutput.size() != getOutputCount())
			return std::vector<float>();
		std::vector<float> err(getOutputCount(), 0);
		for (size_t i = 0; i < getOutputCount(); ++i)
		{
			err[i] = m_backProp.getError(getOutputValue(i), expectedOutput[i]);
		}
		return err;
	}
	float CustomConnectedNeuralNet::getNetError(const std::vector<float>& expectedOutput) const
	{
		if (expectedOutput.size() != getOutputCount())
			return 0;
		float netError = 0;
		for (size_t i = 0; i < getOutputCount(); ++i)
		{
			float diff = m_backProp.getError(getOutputValue(i), expectedOutput[i]);
			netError += diff * diff;
		}
		netError /= getOutputCount();
		return netError;
	}

	void CustomConnectedNeuralNet::removePainter(Visualisation::CustomConnectedNeuralNetPainter* painter)
	{
		auto it = std::find(m_painters.begin(), m_painters.end(), painter);
		if(it != m_painters.end())
			m_painters.erase(it);
	}





}