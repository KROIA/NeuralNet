#pragma once

#include "NeuralNet_base.h"
#include <vector>

namespace NeuralNet
{
	class NEURAL_NET_EXPORT NeuralNetBase
	{
	public:

		NeuralNetBase(unsigned int inputs, unsigned int outputs)
			: m_inputs(inputs)
			, m_outputs(outputs)
		{

		}
		virtual ~NeuralNetBase()
		{}

		unsigned int getInputCount() const
		{
			return m_inputs;
		}
		unsigned int getOutputCount() const
		{
			return m_outputs;
		}

		virtual void setInputValues(const std::vector<float>& values) = 0;
		virtual void setInputValue(unsigned int index, float values) = 0;
		virtual std::vector<float> getOutputValues() const = 0;
		virtual float getOutputValue(unsigned int index) const = 0;
		
		virtual void update() = 0;
	protected:

	private:
		unsigned int m_inputs;
		unsigned int m_outputs;
	};
}