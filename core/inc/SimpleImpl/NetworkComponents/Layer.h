#pragma once

#include "NeuralNet_base.h"
#include "SimpleImpl/NetworkComponents/Neuron.h"
#include "SimpleImpl/NetworkComponents/Connection.h"


namespace NeuralNet
{
	class NEURAL_NET_API Layer
	{
	public:

		std::vector<Neuron*> neurons;
		std::vector<Connection*> inputConnections;
	private:
	};
}