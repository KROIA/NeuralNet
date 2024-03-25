#pragma once

#include "NeuralNet_base.h"
#include <vector>
#include "Base/ActivationFunction.h"
#include "Base/NeuronBase.h"

namespace NeuralNet
{
	class NEURAL_NET_EXPORT Neuron: public NeuronBase
	{
	public:
		Neuron();
		Neuron(Activation::Type &activationType);
		Neuron(const Neuron &neuron);
		Neuron(Neuron&& neuron) noexcept;
		
		Neuron& operator=(const Neuron &neuron);
		Neuron& operator=(Neuron &&neuron) noexcept;

		virtual ~Neuron();

		void setValues(const std::vector<float>& values)
		{
			m_inputValues = values;
		}
		void addInputValue(float value) override
		{
			m_inputValues.push_back(value);
		}
		void clearValue()
		{
			m_inputValues.clear();
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

		void update() override;
	protected:

		using NeuronBase::setNetInput;
		using NeuronBase::setOutput;

		
	private:
		std::vector<float> m_inputValues;
		Activation::ActivationFunction m_activationFunction;
		Activation::Type m_activationType = Activation::Type::sigmoid;
	};
}