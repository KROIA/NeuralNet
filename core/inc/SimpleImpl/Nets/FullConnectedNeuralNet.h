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


		void setInputValues(const std::vector<float>& values) override;
		void setInputValue(unsigned int index, float value) override;
		std::vector<float> getOutputValues() const override;
		float getOutputValue(unsigned int index) const override;

		void update() override;

		std::vector<float> getWeights() const;
		float getWeight(unsigned int layerIdx, unsigned int neuronIdx, unsigned int inputIdx) const;
		void setWeights(const std::vector<float>& weights);
		void setWeight(unsigned int layerIdx, unsigned int neuronIdx, unsigned int inputIdx, float weight);

		Activation::Type getActivationType(unsigned int layerIdx, unsigned int neuronIdx) const;
		void setActivationType(unsigned int layerIdx, unsigned int neuronIdx, Activation::Type type) const;
		void setActivationType(unsigned int layerIdx, Activation::Type type) const;
		void setActivationType(Activation::Type type) const;

		float getInputValue(unsigned int index) const;
		float getHiddenValue(unsigned int layerIdx, unsigned int neuronIdx) const;
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

		std::vector<float> m_inputValues;
		std::vector<float> m_outputValues;
	};
}