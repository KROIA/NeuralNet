#pragma once

#include "NeuralNet_base.h"
#include "CustomConnectedNeuralNet.h"


namespace NeuralNet
{
	namespace Visualisation
	{
		class VisuFullConnectedNeuronalNet;
	}
	
	class NEURAL_NET_API FullConnectedNeuralNet : public CustomConnectedNeuralNet
	{
	public:
		FullConnectedNeuralNet(
			const std::vector<Neuron::ID>& inputNeuronIDs,
			unsigned int hiddenLayerCount, 
			unsigned int hiddenLayerSize,
			const std::vector<Neuron::ID>& outputNeuronIDs);

		~FullConnectedNeuralNet();


	protected:


	private:
	};
}