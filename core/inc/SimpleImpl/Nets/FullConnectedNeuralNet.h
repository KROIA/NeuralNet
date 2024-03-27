#pragma once

#include "NeuralNet_base.h"
#include "Base/NeuralNetBase.h"
#include "SimpleImpl/NetworkComponents/Neuron.h"
#include "SimpleImpl/NetworkComponents/Connection.h"
#include "SimpleImpl/NetworkComponents/Layer.h"

#include "LearnAlgo/Backpropagation.h"


namespace NeuralNet
{
	namespace Visualisation
	{
		class VisuFullConnectedNeuronalNet;
	}
	
	class NEURAL_NET_EXPORT FullConnectedNeuralNet : public NeuralNetBase
	{
	public:
		FullConnectedNeuralNet(
			unsigned int inputSize, 
			unsigned int hiddenLayerCount, 
			unsigned int hiddenLayerSize,
			unsigned int outputSize);

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

		Neuron* getNeuron(Neuron::ID id);

		void update() override;

		size_t getWeightCount() const
		{
			if (m_hiddenLayerCount > 0)
			{
				return (m_hiddenLayerCount - 1) * m_hiddenLayerSize * m_hiddenLayerSize +
					m_hiddenLayerSize * getInputCount() +
					m_hiddenLayerSize * getOutputCount();
			}
			return getInputCount() * getOutputCount();
		}
		std::vector<float> getWeights() const;
		float getWeight(unsigned int layerIdx, unsigned int neuronIDx, unsigned int inputIdx) const;
		void setWeights(const std::vector<float>& weights);
		void setWeight(unsigned int layerIdx, unsigned int neuronIDx, unsigned int inputIdx, float weight);

		Activation::Type getActivationType(unsigned int layerIdx, unsigned int neuronIDx) const;
		void setActivationType(unsigned int layerIdx, unsigned int neuronIDx, Activation::Type type) const;
		void setActivationType(unsigned int layerIdx, Activation::Type type) const;
		void setActivationType(Activation::Type type) const;

		float getInputValue(unsigned int index) const;
		float getHiddenValue(unsigned int layerIdx, unsigned int neuronIDx) const;


		Visualisation::VisuFullConnectedNeuronalNet *createVisualisation();

		void learn(const std::vector<float>& expectedOutput);
		std::vector<float> getOutputError(const std::vector<float>& expectedOutput) const;
		float getNetError(const std::vector<float>& expectedOutput) const;


		/*std::vector<Layer>& getLayers()
		{
			return m_layers;
		}*/
	protected:


	private:
		void buildNetwork();
		void destroyNetwork();

		unsigned int m_hiddenLayerCount;
		unsigned int m_hiddenLayerSize;

		/*struct LayerData
		{
			std::vector<Neuron*> neurons;
			std::vector<Connection*> inputConnections;
		};*/

		std::vector<Layer> m_layers;

		std::vector<float> m_inputValues;
		std::vector<float> m_outputValues;

		std::vector<Visualisation::VisuFullConnectedNeuronalNet*> m_visualisations;
		
		LearnAlgo::Backpropagation m_backProp;
	};
}