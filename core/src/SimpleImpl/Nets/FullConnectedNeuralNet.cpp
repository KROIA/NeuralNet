#include "SimpleImpl/Nets/FullConnectedNeuralNet.h"
#include "SimpleImpl/NetworkComponents/InputNeuron.h"

namespace NeuralNet
{
	FullConnectedNeuralNet::FullConnectedNeuralNet(
		unsigned int inputSize,
		unsigned int hiddenLayerCount,
		unsigned int hiddenLayerSize,
		unsigned int outputSize)
		: CustomConnectedNeuralNet(inputSize, outputSize)
	{
		if (hiddenLayerSize == 0 || hiddenLayerCount == 0)
		{
			hiddenLayerSize = 0;
			hiddenLayerCount = 0;
		}

		std::vector<ConnectionInfo> connections;

		Neuron::ID id = 0;
		// Create input layer
		if (hiddenLayerSize > 0)
		{
			for (unsigned int i = 0; i < inputSize; i++)
			{
				for (unsigned int j = 0; j < hiddenLayerSize; j++)
				{
					ConnectionInfo conInfo;
					conInfo.fromNeuronID = id;
					conInfo.toNeuronID = inputSize + j;
					conInfo.weight = QSFML::Utilities::RandomEngine::getFloat(-1, 1);
					connections.push_back(conInfo);
				}
				id++;
			}
			//id += hiddenLayerSize;

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
				for (unsigned int j = 0; j < outputSize; j++)
				{
					ConnectionInfo conInfo;
					conInfo.fromNeuronID = id + i;
					conInfo.toNeuronID = id + hiddenLayerSize + j;
					conInfo.weight = QSFML::Utilities::RandomEngine::getFloat(-1, 1);
					connections.push_back(conInfo);
				}
			}
		}
		else
		{
			// No hidden layers
			for (unsigned int i = 0; i < inputSize; i++)
			{
				for (unsigned int j = 0; j < outputSize; j++)
				{
					ConnectionInfo conInfo;
					conInfo.fromNeuronID = id;
					conInfo.toNeuronID = inputSize + j;
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