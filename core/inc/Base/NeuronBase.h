#pragma once

#include "NeuralNet_base.h"

namespace NeuralNet
{
	class NEURAL_NET_EXPORT NeuronBase
	{
	public:
		NeuronBase()
		{}
		NeuronBase(const NeuronBase& neuron)
		{
			m_netinput = neuron.m_netinput;
			m_output = neuron.m_output;
		}
		NeuronBase(NeuronBase&& neuron) noexcept
		{
			m_netinput = neuron.m_netinput;
			m_output = neuron.m_output;
		}

		NeuronBase& operator=(const NeuronBase& neuron)
		{
			m_netinput = neuron.m_netinput;
			m_output = neuron.m_output;
			return *this;
		}
		NeuronBase& operator=(NeuronBase&& neuron) noexcept
		{
			m_netinput = neuron.m_netinput;
			m_output = neuron.m_output;
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
	protected:

	private:
		float m_netinput = 0.f;
		float m_output = 0.f;
	};
}