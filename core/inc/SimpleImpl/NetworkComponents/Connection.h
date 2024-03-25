#pragma once

#include "NeuralNet_base.h"
#include "Base/NeuronBase.h"

namespace NeuralNet
{
	class NEURAL_NET_EXPORT Connection
	{
	public:
		Connection()
		{}
		Connection(NeuronBase*from, NeuronBase* to, float weight)
			: m_startNeuron(from)
			, m_endNeuron(to)
			, m_weight(weight)
		{}
		~Connection()
		{}

		void setStartNeuron(NeuronBase* startNeuron)
		{
			m_startNeuron = startNeuron;
		}
		void setEndNeuron(NeuronBase* destinationNeuron)
		{
			m_endNeuron = destinationNeuron;
		}
		void setWeight(float weight)
		{
			m_weight = weight;
		}


		NeuronBase* getStartNeuron() const
		{
			return m_startNeuron;
		}
		NeuronBase* getEndNeuron() const
		{
			return m_endNeuron;
		}

		float getWeight() const
		{
			return m_weight;
		}
		
		void updateWeight(float deltaWeight)
		{
			m_weight += deltaWeight;
		}
		void passValue(float Value) const
		{
			if(m_endNeuron)
				m_endNeuron->addInputValue(getOutputValue(Value));
		}
		void passValue() const
		{
			if (!m_endNeuron || !m_startNeuron)
				return;
			m_endNeuron->addInputValue(getOutputValue());
		}
		float getOutputValue(float inputValue) const
		{
			return inputValue * m_weight;
		}
		float getOutputValue() const
		{
			return getOutputValue(m_startNeuron->getOutput());
		}
	protected:

		
	private:
		NeuronBase* m_startNeuron = nullptr;
		NeuronBase* m_endNeuron = nullptr;
		float m_weight = 1.f;
	};
}