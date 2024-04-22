#pragma once

#include "NeuralNet_base.h"

#include <unordered_map>

#include "SimpleImpl/NetworkComponents/Neuron.h"
#include "SimpleImpl/NetworkComponents/Connection.h"
#include "SimpleImpl/NetworkComponents/Layer.h"

namespace NeuralNet
{
	struct NetworkData
	{
		/// <summary>
		/// ID - Instance pair
		/// </summary>
		std::unordered_map<Neuron::ID, Neuron*> neurons;

		/// <summary>
		/// Sorted by the Neuron::ID
		/// </summary>
		//std::vector<Neuron*> sortedNeurons; 

		/// <summary>
		/// Container for all connections
		/// </summary>
		std::vector<Connection*> connections;

		/// <summary>
		/// Same objects, but splitted into layers
		/// </summary>
		std::vector<Layer> layers;

		unsigned int neuronCount;
	};

	struct ConnectionInfo
	{
		Neuron::ID fromNeuronID;
		Neuron::ID toNeuronID;
		float weight;
	};
}