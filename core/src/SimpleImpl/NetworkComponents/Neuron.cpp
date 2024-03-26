#include "SimpleImpl/NetworkComponents/Neuron.h"
#include "SimpleImpl/NetworkComponents/Connection.h"

namespace NeuralNet
{
	Neuron::Neuron()
		//: Neuron()
	{
		m_activationFunction = Activation::getActivationFunction(m_activationType);
	}
	Neuron::Neuron(Activation::Type& activationType)
		//: Neuron(activationType)
	{
		m_activationType = activationType;
		m_activationFunction = Activation::getActivationFunction(m_activationType);
	}
	Neuron::Neuron(const Neuron& neuron)
		//: Neuron(neuron)
	{
		//m_inputValues = neuron.m_inputValues;
		m_netinput = neuron.m_netinput;
		m_output = neuron.m_output;
		m_activationType = neuron.m_activationType;
		m_activationFunction = Activation::getActivationFunction(m_activationType);
	}
	Neuron::Neuron(Neuron&& neuron) noexcept
		//: Neuron(neuron)
	{
		//m_inputValues = std::move(neuron.m_inputValues);
		m_netinput = neuron.m_netinput;
		m_output = neuron.m_output;
		m_activationType = neuron.m_activationType;
		m_activationFunction = Activation::getActivationFunction(m_activationType);
	}
	
	Neuron& Neuron::operator=(const Neuron& neuron)
	{
		//Neuron::operator=(neuron);
		//m_inputValues = neuron.m_inputValues;
		m_netinput = neuron.m_netinput;
		m_output = neuron.m_output;
		m_activationType = neuron.m_activationType;
		m_activationFunction = Activation::getActivationFunction(m_activationType);
		return *this;
	}
	Neuron& Neuron::operator=(Neuron&& neuron) noexcept
	{
		//Neuron::operator=(neuron);
		//m_inputValues = std::move(neuron.m_inputValues);
		m_netinput = neuron.m_netinput;
		m_output = neuron.m_output;
		m_activationType = neuron.m_activationType;
		m_activationFunction = Activation::getActivationFunction(m_activationType);
		return *this;
	}
	Neuron::~Neuron()
	{
		auto copy = m_inputConnections;
		for (auto& conn : copy)
			conn->setEndNeuron(nullptr);
	}

	void Neuron::update()
	{
		float netinput = 0;
		/*for (auto& signal : m_inputValues)
		{
			netinput += signal;
		}*/
		for (auto& conn : m_inputConnections)
		{
			netinput += conn->getOutputValue();
		}
		setNetInput(netinput);
		setOutput(getActivationFunction()(netinput));
	}
}