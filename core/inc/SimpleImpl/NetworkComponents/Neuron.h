#pragma once

#include "NeuralNet_base.h"
#include <vector>
#include "Base/ActivationFunction.h"
//#include "Base/Neuron.h"

namespace NeuralNet
{
	class Connection;
	class NEURAL_NET_EXPORT Neuron
	{
	public:
		Neuron();
		Neuron(Activation::Type &activationType);
		Neuron(const Neuron &neuron);
		Neuron(Neuron&& neuron) noexcept;
		
		Neuron& operator=(const Neuron &neuron);
		Neuron& operator=(Neuron &&neuron) noexcept;

		virtual ~Neuron();

		bool addInputConnection(Connection* connection)
		{
			for (auto& conn : m_inputConnections)
			{
				if(conn == connection)
					return false;
			}
			m_inputConnections.push_back(connection);
			return true;
		}
		bool removeInputConnection(Connection* connection)
		{
			for (auto it = m_inputConnections.begin(); it != m_inputConnections.end(); ++it)
			{
				if (*it == connection)
				{
					m_inputConnections.erase(it);
					return true;
				}
			}
			return true;
		}
		const std::vector<Connection*>& getInputConnections() const
		{
			return m_inputConnections;
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

		/*void setInputValues(const std::vector<float>& values)
		{
			m_inputValues = values;
		}
		void addInputValue(float value) 
		{
			m_inputValues.push_back(value);
		}
		void clearValue()
		{
			m_inputValues.clear();
		}*/
		

		virtual void update();

		Activation::ActivationFunction getActivationFunction() const
		{
			return m_activationFunction;
		}
		Activation::ActivationFunction getActivationFunctionDerivetive() const
		{
			return Activation::getActivationDerivetiveFunction(m_activationType);
		}

		float getOutput() const
		{
			return m_output;
		}
		float getNetInput() const
		{
			return m_netinput;
		}
	protected:

		void setOutput(float output)
		{
			m_output = output;
		}
		void setNetInput(float netinput)
		{
			m_netinput = netinput;
		}

		
		//std::vector<float> m_inputValues;
		std::vector<Connection*> m_inputConnections;
		float m_netinput = 0.f;
		float m_output = 0.f;
		Activation::Type m_activationType = Activation::Type::linear;
		Activation::ActivationFunction m_activationFunction;

	};
}