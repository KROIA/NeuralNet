#pragma once

#include "NeuralNet_base.h"
#include "Neuron.h"
#include "Connection.h"

namespace NeuralNet
{
	class NEURAL_NET_EXPORT InputNeuron : public Neuron
	{
	public:
		InputNeuron(ID id)
			: Neuron(id)
		{
			m_activationType = Activation::Type::linear;
			m_activationFunction = Activation::getActivationFunction(m_activationType);
		}

		void setActivationType(Activation::Type type) override
		{
			// Do nothing
			NN_UNUSED(type);
		}

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