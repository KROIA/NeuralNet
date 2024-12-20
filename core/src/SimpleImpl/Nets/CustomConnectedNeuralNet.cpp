#include "SimpleImpl/Nets/CustomConnectedNeuralNet.h"
#include "SimpleImpl/NetworkComponents/InputNeuron.h"
#include "Visualisation/CustomConnectedNeuralNetPainter.h"
#include <iostream>

namespace NeuralNet
{
	Log::LogObject& CustomConnectedNeuralNet::getLogger()
	{
		static Log::LogObject logger("CustomConnectedNeuralNet");
		return logger;
	}
	CustomConnectedNeuralNet::CustomConnectedNeuralNet(
		const std::vector<Neuron::ID>& inputNeuronIDs,
		const std::vector<Neuron::ID>& outputNeuronIDs)
		: NeuralNetBase(inputNeuronIDs.size(), outputNeuronIDs.size())
		, m_inputNeuronIDs(inputNeuronIDs)
		, m_outputNeuronIDs(outputNeuronIDs)
	{
		m_inputValues = std::vector<float>(inputNeuronIDs.size(), 0);
		m_outputValues = std::vector<float>(outputNeuronIDs.size(), 0);
	}

	CustomConnectedNeuralNet::~CustomConnectedNeuralNet()
	{
		std::vector<Visualisation::CustomConnectedNeuralNetPainter*> painters = m_painters;
		m_painters.clear();
		for (auto painter : painters)
		{
			QSFML::Internal::LifetimeChecker::deleteSecured(painter);
		}
		destroyNetwork();
	}


	void CustomConnectedNeuralNet::addConnection(const ConnectionInfo& connectionInfo)
	{
		m_buildingConnections.push_back(connectionInfo);
		needsStructureUpdate();
	}
	void CustomConnectedNeuralNet::addConnection(Neuron::ID fromNeuronID, Neuron::ID toNeuronID)
	{
		addConnection(fromNeuronID, toNeuronID, QSFML::Utilities::RandomEngine::getFloat(-1, 1));
	}
	void CustomConnectedNeuralNet::addConnection(Neuron::ID fromNeuronID, Neuron::ID toNeuronID, float weight)
	{
		ConnectionInfo info;
		info.fromNeuronID = fromNeuronID;
		info.toNeuronID = toNeuronID;
		info.weight = weight;
		addConnection(info);
	}
	void CustomConnectedNeuralNet::setConnections(const std::vector<ConnectionInfo>& connections)
	{
		m_buildingConnections = connections;
		needsStructureUpdate();
	}
	void CustomConnectedNeuralNet::removeConnection(Neuron::ID fromNeuronID, Neuron::ID toNeuronID)
	{
		for (auto it = m_buildingConnections.begin(); it != m_buildingConnections.end(); ++it)
		{
			if (it->fromNeuronID == fromNeuronID && it->toNeuronID == toNeuronID)
			{
				m_buildingConnections.erase(it);
				needsStructureUpdate();
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
			m_biasList,
			m_activationTypes, 
			m_defaultLayerActivationTypes,
			m_defaultActivationType,
			m_inputNeuronIDs,
			m_outputNeuronIDs,
			m_networkData);
		for (auto painter : m_painters)
		{
			painter->buildNetwork();
		}
		m_networkStructureOutOfDate = false;
	}
	void CustomConnectedNeuralNet::destroyNetwork()
	{
		needsStructureUpdate();
		CustomConnectedNeuralNetBuilder::destroyNetwork(m_networkData);
	}

	void CustomConnectedNeuralNet::setInputValues(const std::vector<float>& values)
	{
		if (values.size() != getInputSize())
		{
			size_t minSize = std::min(values.size(), (size_t)getInputSize());
			std::copy(values.begin(), values.begin() + minSize, m_inputValues.begin());
		}
		else
		{
			m_inputValues = values;
		}
		for (size_t i = 0; i < m_inputValues.size(); ++i)
		{
			if(isnan(m_inputValues[i]) || isinf(m_inputValues[i]))
				m_inputValues[i] = 0.0f;
		}
	}
	std::vector<float> CustomConnectedNeuralNet::getInputValues() const
	{
		return m_inputValues;
	}
	float CustomConnectedNeuralNet::getInputValue(unsigned int index) const
	{
		if(index < m_inputValues.size())
			return m_inputValues[index];
		return 0.0f;
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
	const Neuron* CustomConnectedNeuralNet::getNeuron(Neuron::ID id) const
	{
		auto it = m_networkData.neurons.find(id);
		if (it != m_networkData.neurons.end())
		{
			return it->second;
		}
		return nullptr;
	}
	Neuron* CustomConnectedNeuralNet::getNeuron(unsigned int layerIdx, unsigned int neuronIdx)
	{
		if (m_networkData.layers.size() <= layerIdx)
			return nullptr;
		const Layer& layer = m_networkData.layers[layerIdx];
		if (layer.neurons.size() <= neuronIdx)
			return nullptr;
		return layer.neurons[neuronIdx];
	}
	const Neuron* CustomConnectedNeuralNet::getNeuron(unsigned int layerIdx, unsigned int neuronIdx) const
	{
		if (m_networkData.layers.size() <= layerIdx)
			return nullptr;
		const Layer& layer = m_networkData.layers[layerIdx];
		if (layer.neurons.size() <= neuronIdx)
			return nullptr;
		return layer.neurons[neuronIdx];
	}
	std::unordered_map<Neuron::ID, Neuron*> CustomConnectedNeuralNet::getNeurons() const
	{
		return m_networkData.neurons;
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
		else
			needsStructureUpdate();
		m_activationTypes[id] = type;
	}
	void CustomConnectedNeuralNet::setLayerActivationType(unsigned int layerIdx, Activation::Type type)
	{
		m_defaultLayerActivationTypes[layerIdx] = type;
		if (layerIdx < m_networkData.layers.size())
		{
			for (auto& neuron : m_networkData.layers[layerIdx].neurons)
			{
				neuron->setActivationType(type);
				m_activationTypes[neuron->getID()] = type;
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
		if (m_networkStructureOutOfDate)
		{
			std::cout << "Network structure not updated, call CustomConnectedNeuralNet::buildNetwork() first\n";
			return;
		}
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
		m_softMaxSum = 0;
		for (unsigned int i = 0; i < outputLayer.neurons.size(); ++i)
		{
			if (m_enableSoftMax)
			{
				m_outputValues[i] = std::exp(outputLayer.neurons[i]->getOutput());
				m_softMaxSum += m_outputValues[i];
			}
			else
			{
				m_outputValues[i] = outputLayer.neurons[i]->getOutput();
			}
		}
		if (m_enableSoftMax && m_softMaxSum > 0)
		{
			for (unsigned int i = 0; i < outputLayer.neurons.size(); ++i)
			{
				m_outputValues[i] /= m_softMaxSum;
			}
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
		const Neuron* n = getNeuron(layerIdx, neuronIdx);
		if (!n)
			return 0.0f;
		const auto &conn = n->getInputConnections();
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
		Neuron* n = getNeuron(layerIdx, neuronIdx);
		if (!n)
			return;
		const auto& conn = n->getInputConnections();
		if (conn.size() <= inputIdx)
			return;
		conn[inputIdx]->setWeight(weight);
	}

	float CustomConnectedNeuralNet::getBias(unsigned int layerIdx, unsigned int neuronIdx) const
	{
		const Neuron* n = getNeuron(layerIdx, neuronIdx);
		if (!n)
			return 0.0f;
		return n->getBias();
	}
	float CustomConnectedNeuralNet::getBias(Neuron::ID id) const
	{
		const Neuron* n = getNeuron(id);
		if (!n)
			return 0.0f;
		return n->getBias();
	}
	std::vector<float> CustomConnectedNeuralNet::getBias() const
	{
		size_t neuronCount;
		std::vector<float> biasList(m_networkData.neuronCount, 0);

		size_t index = 0;
		for(size_t i=1; i< m_networkData.layers.size(); ++i)
		{
			for (auto& neuron : m_networkData.layers[i].neurons)
			{
				biasList[index] = neuron->getBias();
				++index;
			}
		}

		return biasList;
	}

	void CustomConnectedNeuralNet::setBias(Neuron::ID id, float bias)
	{
		m_biasList[id] = bias;
		Neuron* n = getNeuron(id);
		if (!n)
		{
			getLogger().logError("setBias(...): Invalid neuron id");
			return;
		}
		n->setBias(bias);
	}
	void CustomConnectedNeuralNet::setBias(unsigned int layerIdx, unsigned int neuronIdx, float bias)
	{
		Neuron* n = getNeuron(layerIdx, neuronIdx);
		if (!n)
		{
			getLogger().logError("setBias(...): Invalid layer/neuron index");
			return;
		}
		n->setBias(bias);
		m_biasList[n->getID()] = bias;
	}
	void CustomConnectedNeuralNet::setBias(const std::vector<float>& biasList)
	{
		if (biasList.size() != m_networkData.neuronCount)
		{
			getLogger().logError("setBias(...): Invalid bias list size");
			return; // invalid size
		}

		size_t index = 0;
		for (size_t i = 1; i < m_networkData.layers.size(); ++i)
		{
			for (auto& neuron : m_networkData.layers[i].neurons)
			{
				neuron->setBias(biasList[index]);
				m_biasList[neuron->getID()] = biasList[index];
				++index;
			}
		}
	}
	void CustomConnectedNeuralNet::setBias(float biasForAll)
	{
		for (auto& layer : m_networkData.layers)
		{
			for (auto& neuron : layer.neurons)
			{
				neuron->setBias(biasForAll);
				m_biasList[neuron->getID()] = biasForAll;
			}
		}
	}

	size_t CustomConnectedNeuralNet::getGenomSize() const
	{
		size_t size = 0;
		for (size_t i = 1; i < m_networkData.layers.size(); ++i)
		{
			for (auto& neuron : m_networkData.layers[i].neurons)
			{
				size += 1 + neuron->getInputConnections().size();
			}
		}
		return size;
	}

	std::vector<float> CustomConnectedNeuralNet::getGenom() const
	{
		std::vector<float> genom;
		for (size_t i=1; i<m_networkData.layers.size(); ++i)
		{
			for (auto& neuron : m_networkData.layers[i].neurons)
			{
				genom.push_back(neuron->getBias());
				for (auto& connection : neuron->getInputConnections())
				{
					genom.push_back(connection->getWeight());
				}
			}
		}
		return genom;
	}
	void CustomConnectedNeuralNet::setGenom(const std::vector<float>& genom)
	{
		if (genom.size() != getGenomSize())
		{
			getLogger().logError("setGenom(...): Invalid genom size");
			return;
		}
		size_t index = 0;
		for (size_t i = 1; i < m_networkData.layers.size(); ++i)
		{
			for (auto& neuron : m_networkData.layers[i].neurons)
			{
				neuron->setBias(genom[index++]);
				for (auto& connection : neuron->getInputConnections())
				{
					connection->setWeight(genom[index++]);
				}
			}
		}
	}

	void CustomConnectedNeuralNet::enableNormalizedNetInput(bool enable)
	{
		for(auto& neuron : m_networkData.neurons)
		{
			neuron.second->enableNormalizedNetinput(enable);
		}
	}

	/*
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
			float outp = getOutputValue(i);
			float expected = expectedOutput[i];
			float diff = m_backProp.getError(outp, expected);
			netError += std::abs(diff);
		}
		netError /= getOutputCount();
		return netError;
	}
	*/
	void CustomConnectedNeuralNet::removePainter(Visualisation::CustomConnectedNeuralNetPainter* painter)
	{
		auto it = std::find(m_painters.begin(), m_painters.end(), painter);
		if(it != m_painters.end())
			m_painters.erase(it);
	}





}