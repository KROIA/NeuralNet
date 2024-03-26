#pragma once

#include "NeuralNet_base.h"
#include "Neuron.h"
#include "Connection.h"

namespace NeuralNet
{
	class NEURAL_NET_EXPORT InputNeuron : public Neuron
	{
	public:

		void setValue(float value)
		{
			setNetInput(value);
			setOutput(value);
		}

		void update() override
		{
			
		}
	};
}