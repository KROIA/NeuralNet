#include "SimpleImpl/NetworkComponents/Neuron.h"

namespace NeuralNet
{
	Neuron::Neuron()
		: NeuronBase()
	{
		m_activationFunction = Activation::getActivationFunction(m_activationType);
	}
	Neuron::Neuron(Activation::Type& activationType)
		: NeuronBase()
	{
		m_activationType = activationType;
		m_activationFunction = Activation::getActivationFunction(m_activationType);
	}
	Neuron::Neuron(const Neuron& neuron)
		: NeuronBase(neuron)
	{
		m_activationType = neuron.m_activationType;
		m_activationFunction = Activation::getActivationFunction(m_activationType);
		m_inputValues = neuron.m_inputValues;
	}
	Neuron::Neuron(Neuron&& neuron) noexcept
		: NeuronBase(neuron)
	{
		m_activationType = neuron.m_activationType;
		m_activationFunction = Activation::getActivationFunction(m_activationType);
		m_inputValues = std::move(neuron.m_inputValues);
	}
	
	Neuron& Neuron::operator=(const Neuron& neuron)
	{
		NeuronBase::operator=(neuron);
		m_activationType = neuron.m_activationType;
		m_activationFunction = Activation::getActivationFunction(m_activationType);
		m_inputValues = neuron.m_inputValues;
		return *this;
	}
	Neuron& Neuron::operator=(Neuron&& neuron) noexcept
	{
		NeuronBase::operator=(neuron);
		m_activationType = neuron.m_activationType;
		m_activationFunction = Activation::getActivationFunction(m_activationType);
		m_inputValues = std::move(neuron.m_inputValues);
		return *this;
	}
	Neuron::~Neuron()
	{

	}

	void Neuron::update()
	{
		float netinput = 0;
		for(auto &signal : m_inputValues)
		{
			netinput += signal;
		}
		setNetInput(netinput);
		setOutput(m_activationFunction(netinput));
	}
}