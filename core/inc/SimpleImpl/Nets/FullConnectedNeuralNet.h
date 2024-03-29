#pragma once

#include "NeuralNet_base.h"
#include "CustomConnectedNeuralNet.h"


namespace NeuralNet
{
	namespace Visualisation
	{
		class VisuFullConnectedNeuronalNet;
	}
	
	class NEURAL_NET_EXPORT FullConnectedNeuralNet : public CustomConnectedNeuralNet
	{
	public:
		FullConnectedNeuralNet(
			unsigned int inputSize, 
			unsigned int hiddenLayerCount, 
			unsigned int hiddenLayerSize,
			unsigned int outputSize);

		~FullConnectedNeuralNet();


	protected:


	private:
	};
}