#include "SimpleImpl/NetworkComponents/Connection.h"
#include "SimpleImpl/NetworkComponents/Neuron.h"

namespace NeuralNet
{

	Connection::~Connection()
	{
		//if (m_startNeuron)
		//	m_startNeuron->removeOutputConnection(this);
		if (m_endNeuron)
			m_endNeuron->removeInputConnection(this);
	
	}

	void Connection::setEndNeuron(Neuron* destinationNeuron)
	{
		if (m_endNeuron)
			m_endNeuron->removeInputConnection(this);
		m_endNeuron = destinationNeuron;
		if(m_endNeuron)
			m_endNeuron->addInputConnection(this);
	}
}