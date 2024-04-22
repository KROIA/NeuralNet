#include "SimpleImpl/NetworkComponents/Neuron.h"
#include "SimpleImpl/NetworkComponents/Connection.h"

#include "Visualisation/NeuronPainter.h"

namespace NeuralNet
{
	Neuron::Neuron(ID id)
		: m_id(id)
	{
		m_activationFunction = Activation::getActivationFunction(m_activationType);
		m_bias = 0;
	}
	Neuron::Neuron(ID id, Activation::Type& activationType)
		: m_id(id)
	{
		m_activationType = activationType;
		m_activationFunction = Activation::getActivationFunction(m_activationType);
		m_bias = 0;
	}
	Neuron::Neuron(const Neuron& neuron)
		: m_id(neuron.m_id)
	{
		m_netinput = neuron.m_netinput;
		m_output = neuron.m_output;
		m_activationType = neuron.m_activationType;
		m_activationFunction = Activation::getActivationFunction(m_activationType);
		m_bias = neuron.m_bias;
	}
	Neuron::Neuron(Neuron&& neuron) noexcept
		: m_id(neuron.m_id)
	{
		m_netinput = neuron.m_netinput;
		m_output = neuron.m_output;
		m_activationType = neuron.m_activationType;
		m_activationFunction = Activation::getActivationFunction(m_activationType);
		m_bias = neuron.m_bias;
	}
	
	Neuron& Neuron::operator=(const Neuron& neuron)
	{
		m_id = neuron.m_id;
		m_netinput = neuron.m_netinput;
		m_output = neuron.m_output;
		m_activationType = neuron.m_activationType;
		m_activationFunction = Activation::getActivationFunction(m_activationType);
		m_bias = neuron.m_bias;
		return *this;
	}
	Neuron& Neuron::operator=(Neuron&& neuron) noexcept
	{
		m_id = neuron.m_id;
		m_netinput = neuron.m_netinput;
		m_output = neuron.m_output;
		m_activationType = neuron.m_activationType;
		m_activationFunction = Activation::getActivationFunction(m_activationType);
		m_bias = neuron.m_bias;
		return *this;
	}
	Neuron::~Neuron()
	{
		auto copy = m_inputConnections;
		for (auto& conn : copy)
			conn->setEndNeuron(nullptr);

		auto copyPainter = m_painters;
		m_painters.clear();
		for (auto& painter : copyPainter)
			delete painter;
	}

	void Neuron::update()
	{
		float netinput = m_bias;
		for (auto& conn : m_inputConnections)
		{
			netinput += conn->getOutputValue();
		}
		setNetInput(netinput);
		setOutput(getActivationFunction()(netinput));
	}

	Visualisation::NeuronPainter* Neuron::createVisualisation()
	{
		Visualisation::NeuronPainter* painter = new Visualisation::NeuronPainter(this);
		m_painters.push_back(painter);
		return painter;
	}
}