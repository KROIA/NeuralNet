#pragma once

#include "NeuralNet_base.h"
#include "Base/ActivationFunction.h"

namespace NeuralNet
{
	class NEURAL_NET_EXPORT NeuronBase
	{
	public:
		NeuronBase()
		{
			m_activationFunction = Activation::getActivationFunction(m_activationType);
		}
		NeuronBase(Activation::Type& activationType)
		{
			m_activationType = activationType;
			m_activationFunction = Activation::getActivationFunction(m_activationType);
		}
		NeuronBase(const NeuronBase& neuron)
		{
			m_netinput = neuron.m_netinput;
			m_output = neuron.m_output;
			m_activationType = neuron.m_activationType;
			m_activationFunction = Activation::getActivationFunction(m_activationType);
		}
		NeuronBase(NeuronBase&& neuron) noexcept
		{
			m_netinput = neuron.m_netinput;
			m_output = neuron.m_output;
			m_activationType = neuron.m_activationType;
			m_activationFunction = Activation::getActivationFunction(m_activationType);
		}

		NeuronBase& operator=(const NeuronBase& neuron)
		{
			m_netinput = neuron.m_netinput;
			m_output = neuron.m_output;
			m_activationType = neuron.m_activationType;
			m_activationFunction = Activation::getActivationFunction(m_activationType);
			return *this;
		}
		NeuronBase& operator=(NeuronBase&& neuron) noexcept
		{
			m_netinput = neuron.m_netinput;
			m_output = neuron.m_output;
			m_activationType = neuron.m_activationType;
			m_activationFunction = Activation::getActivationFunction(m_activationType);
			return *this;
		}
		virtual ~NeuronBase()
		{}


		void setOutput(float output)
		{
			m_output = output;
		}
		void setNetInput(float netinput)
		{
			m_netinput = netinput;
		}

		virtual void addInputValue(float value)
		{
			m_netinput = value;
		}

		float getNetInput() const
		{
			return m_netinput;
		}
		float getOutput() const
		{
			return m_output;
		}

		virtual void update()
		{
			m_output = m_netinput;
		}

		void setActivationType(Activation::Type type)
		{
			m_activationType = type;
			m_activationFunction = Activation::getActivationFunction(type);
		}
		Activation::Type getActivationType() const
		{
			return m_activationType;
		}

		virtual void clearValue()
		{

		}
	protected:

		Activation::ActivationFunction getActivationFunction() const
		{
			return m_activationFunction;
		}

	private:
		float m_netinput = 0.f;
		float m_output = 0.f;
		Activation::Type m_activationType = Activation::Type::linear;
		Activation::ActivationFunction m_activationFunction;
	};
}