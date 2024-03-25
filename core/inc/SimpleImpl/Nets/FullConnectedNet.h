#pragma once

#include "NeuralNet_base.h"
#include "Base/NeuralNetBase.h"
#include "SimpleImpl/NetworkComponents/Neuron.h"
#include "SimpleImpl/NetworkComponents/Connection.h"


namespace NeuralNet
{

	class NEURAL_NET_EXPORT FullConnectedNeuralNet : public NeuralNetBase
	{
	public:
		FullConnectedNeuralNet(
			unsigned int inputSize, 
			unsigned int outputSize, 
			unsigned int hiddenLayerCount, 
			unsigned int hiddenLayerSize);

		~FullConnectedNeuralNet();


		unsigned int getHiddenLayerCount() const
		{
			return m_hiddenLayerCount;
		}
		unsigned int getHiddenLayerSize() const
		{
			return m_hiddenLayerSize;
		}


		void setInputValues(const std::vector<float>& signals_) override;
		void setInputValue(unsigned int index, float signal) override;
		std::vector<float> getOutputValues() const override;
		float getOutputValue(unsigned int index) const override;

		void update() override;
	protected:


	private:
		void buildNetwork();
		void destroyNetwork();

		unsigned int m_hiddenLayerCount;
		unsigned int m_hiddenLayerSize;

		struct LayerData
		{
			std::vector<NeuronBase*> neurons;
			std::vector<Connection*> inputConnections;
		};

		std::vector<LayerData> m_layers;

		std::vector<float> m_inputSignals;
		std::vector<float> m_outputSignals;
	};
}