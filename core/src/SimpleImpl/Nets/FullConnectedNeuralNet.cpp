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

		Neuron::ID id = inputNeuronIDs.size()+outputNeuronIDs.size();
		// Create input layer
		if (hiddenLayerSize > 0)
		{
			for (unsigned int i = 0; i < inputNeuronIDs.size(); i++)
			{
				for (unsigned int j = 0; j < hiddenLayerSize; j++)
				{
					ConnectionInfo conInfo;
					conInfo.fromNeuronID = inputNeuronIDs[i];
					conInfo.toNeuronID = id + j;
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
						conInfo.fromNeuronID = id + j;
						conInfo.toNeuronID = id + hiddenLayerSize + k;
						conInfo.weight = QSFML::Utilities::RandomEngine::getFloat(-1, 1);
						connections.push_back(conInfo);
					}
				}
				id += hiddenLayerSize;
			}
			//id -= hiddenLayerSize;

			// Create output layer
			for (unsigned int i = 0; i < hiddenLayerSize; i++)
			{
				for (unsigned int j = 0; j < outputNeuronIDs.size(); j++)
				{
					ConnectionInfo conInfo;
					conInfo.fromNeuronID = id + i;
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
					conInfo.fromNeuronID = id;
					conInfo.toNeuronID = outputNeuronIDs[j];
					conInfo.weight = QSFML::Utilities::RandomEngine::getFloat(-1, 1);
					connections.push_back(conInfo);
				}
				id++;
			}
		}

		setConnections(connections);
		buildNetwork();
	}

	FullConnectedNeuralNet::~FullConnectedNeuralNet()
	{

	}



}