#include "SimpleImpl/Nets/FullConnectedNeuralNet.h"
#include "SimpleImpl/NetworkComponents/InputNeuron.h"

namespace NeuralNet
{
	FullConnectedNeuralNet::FullConnectedNeuralNet(
		const std::vector<Neuron::ID>& inputNeuronIDs,
		unsigned int hiddenLayerCount,
		unsigned int hiddenLayerSize,
		const std::vector<Neuron::ID>& outputNeuronIDs)
		: CustomConnectedNeuralNet(inputNeuronIDs, outputNeuronIDs)
	{
		if (hiddenLayerSize == 0 || hiddenLayerCount == 0)
		{
			hiddenLayerSize = 0;
			hiddenLayerCount = 0;
		}

		std::vector<ConnectionInfo> connections;

		std::unordered_map<Neuron::ID, Neuron::ID> usedIDs;
		for (unsigned int i = 0; i < inputNeuronIDs.size(); i++)
		{
			usedIDs[inputNeuronIDs[i]] = inputNeuronIDs[i];
		}
		for (unsigned int i = 0; i < outputNeuronIDs.size(); i++)
		{
			usedIDs[outputNeuronIDs[i]] = outputNeuronIDs[i];
		}

		std::vector<std::vector<Neuron::ID>> hiddenLayerIDs;
		{
			Neuron::ID id = inputNeuronIDs.size() + outputNeuronIDs.size();
			while (usedIDs.find(id) != usedIDs.end()) { id++; }
			for (unsigned int i = 0; i < hiddenLayerCount; i++)
			{
				std::vector<Neuron::ID> layer;
				for (unsigned int j = 0; j < hiddenLayerSize; j++)
				{
					layer.push_back(id);
					usedIDs[id] = id;
					id++;
					while (usedIDs.find(id) != usedIDs.end()) { id++; }
				}
				hiddenLayerIDs.push_back(layer);
			}
		}
		
		// Create input layer
		if (hiddenLayerSize > 0)
		{
			for (unsigned int i = 0; i < inputNeuronIDs.size(); i++)
			{
				for (unsigned int j = 0; j < hiddenLayerSize; j++)
				{
					ConnectionInfo conInfo;
					conInfo.fromNeuronID = inputNeuronIDs[i];
					conInfo.toNeuronID = hiddenLayerIDs[0][j];
					conInfo.weight = QSFML::Utilities::RandomEngine::getFloat(-1, 1);
					connections.push_back(conInfo);
				}
			}

			// Create hidden layers
			for (unsigned int i = 0; i < hiddenLayerCount - 1; i++)
			{
				for (unsigned int j = 0; j < hiddenLayerSize; j++)
				{
					for (unsigned int k = 0; k < hiddenLayerSize; k++)
					{
						ConnectionInfo conInfo;
						conInfo.fromNeuronID = hiddenLayerIDs[i][k]; 
						conInfo.toNeuronID = hiddenLayerIDs[i+1][j]; 
						conInfo.weight = QSFML::Utilities::RandomEngine::getFloat(-1, 1);
						connections.push_back(conInfo);
					}
				}
			}
			//id -= hiddenLayerSize;

			// Create output layer
			for (unsigned int i = 0; i < hiddenLayerSize; i++)
			{
				for (unsigned int j = 0; j < outputNeuronIDs.size(); j++)
				{
					ConnectionInfo conInfo;
					conInfo.fromNeuronID = hiddenLayerIDs[hiddenLayerIDs.size()-1][i];
					conInfo.toNeuronID = outputNeuronIDs[j];
					conInfo.weight = QSFML::Utilities::RandomEngine::getFloat(-1, 1);
					connections.push_back(conInfo);
				}
			}
		}
		else
		{
			// No hidden layers
			for (unsigned int i = 0; i < inputNeuronIDs.size(); i++)
			{
				for (unsigned int j = 0; j < outputNeuronIDs.size(); j++)
				{
					ConnectionInfo conInfo;
					conInfo.fromNeuronID = inputNeuronIDs[i];
					conInfo.toNeuronID = outputNeuronIDs[j];
					conInfo.weight = QSFML::Utilities::RandomEngine::getFloat(-1, 1);
					connections.push_back(conInfo);
				}
			}
		}


		for (auto& id : outputNeuronIDs)
		{
			m_biasList.insert({ id, QSFML::Utilities::RandomEngine::getFloat(-1, 1) });
		}
		for (auto& layer : hiddenLayerIDs)
		{
			for (auto& id : layer)
			{
				m_biasList.insert({ id, QSFML::Utilities::RandomEngine::getFloat(-1, 1) });
			}
		}
		
		

		setConnections(connections);
		buildNetwork();
	}

	FullConnectedNeuralNet::~FullConnectedNeuralNet()
	{

	}



}