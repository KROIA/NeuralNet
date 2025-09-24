#pragma once

#include "NeuralNet_base.h"
#include "Base/NeuralNetBase.h"
#include "SimpleImpl/Nets/NetworkData.h"
#include "SimpleImpl/NetworkComponents/Neuron.h"
#include "SimpleImpl/NetworkComponents/Connection.h"
#include "SimpleImpl/NetworkComponents/Layer.h"
#include "QSFML_EditorWidget.h"
#include <unordered_map>

//#include "LearnAlgo/Backpropagation.h"

namespace NeuralNet
{
	namespace Visualisation
	{
		class CustomConnectedNeuralNetPainter;
	}
	namespace LearnAlgo
	{
		class Backpropagation;
	}
	class NEURAL_NET_API CustomConnectedNeuralNet : public NeuralNetBase
	{
		friend class Visualisation::CustomConnectedNeuralNetPainter;
		friend LearnAlgo::Backpropagation;
	public:
		static Log::LogObject& getLogger();
		CustomConnectedNeuralNet(
			const std::vector<Neuron::ID>& inputNeuronIDs,
			const std::vector<Neuron::ID>& outputNeuronIDs);

		~CustomConnectedNeuralNet();

		
		void clearConnections()
		{
			m_buildingConnections.clear();
			m_activationTypes.clear();
			clearBiasList();
		}
		void addConnection(const ConnectionInfo& connectionInfo);
		void addConnection(Neuron::ID fromNeuronID, Neuron::ID toNeuronID);
		void addConnection(Neuron::ID fromNeuronID, Neuron::ID toNeuronID, float weight);
		void setConnections(const std::vector<ConnectionInfo>& connections);
		void removeConnection(Neuron::ID fromNeuronID, Neuron::ID toNeuronID);
		std::vector<ConnectionInfo> getConnections() const;
		size_t getWeightCount() const
		{
			return m_networkData.connections.size();
		}
		void clearBiasList()
		{
			m_biasList.clear();
		}
		void enableSoftMaxOutput(bool enable)
		{
			m_enableSoftMax = enable;
		}
		bool isSoftMaxOutputEnabled() const
		{
			return m_enableSoftMax;
		}

		void buildNetwork();
		void destroyNetwork();

		void setInputValues(const std::vector<float>& values) override;
		void setInputValue(unsigned int index, float values) override;
		std::vector<float> getInputValues() const override;
		float getInputValue(unsigned int index) const override;
		std::vector<float> getOutputValues() const override;
		float getOutputValue(unsigned int index) const override;

		Activation::Type getActivationType(Neuron::ID id) const;
		void setActivationType(Neuron::ID id, Activation::Type type);
		void setLayerActivationType(unsigned int layerIdx, Activation::Type type);
		void setActivationType(Activation::Type type);

		Neuron* getNeuron(Neuron::ID id);
		const Neuron* getNeuron(Neuron::ID id) const;
		Neuron* getNeuron(unsigned int layerIdx, unsigned int neuronIdx);
		const Neuron* getNeuron(unsigned int layerIdx, unsigned int neuronIdx) const;
		std::unordered_map<Neuron::ID, Neuron*> getNeurons() const;

		//std::unordered_map<Neuron::ID, Neuron*> getNeurons_

		void update() override;

		Visualisation::CustomConnectedNeuralNetPainter* createVisualisation();

		std::vector<float> getWeights() const;
		float getWeight(unsigned int layerIdx, unsigned int neuronIdx, unsigned int inputIdx) const;
		void setWeights(const std::vector<float>& weights);
		void setWeight(unsigned int layerIdx, unsigned int neuronIdx, unsigned int inputIdx, float weight);

		float getBias(unsigned int layerIdx, unsigned int neuronIdx) const;
		float getBias(Neuron::ID id) const;
		std::vector<float> getBias() const;
		void setBias(Neuron::ID id, float bias);
		void setBias(unsigned int layerIdx, unsigned int neuronIdx, float bias);
		void setBias(const std::vector<float>& biasList);
		void setBias(float biasForAll);

		size_t getGenomSize() const;
		std::vector<float> getGenom() const;
		void setGenom(const std::vector<float>& genom);

		void enableNormalizedNetInput(bool enable);

		//void setLearningRate(float learnRate){ m_backProp.setLearningRate(learnRate); }
		//void learn(const std::vector<float>& expectedOutput);
		//std::vector<float> getOutputError(const std::vector<float>& expectedOutput) const;
		//float getNetError(const std::vector<float>& expectedOutput) const;


		unsigned int getInputLayerIndex() const
		{
			return 0;
		}
		unsigned int getOutputLayerIndex() const
		{
			return m_networkData.layers.size() - 1;
		}

	protected:
		void removePainter(Visualisation::CustomConnectedNeuralNetPainter* painter);
		void needsStructureUpdate()
		{
			m_networkStructureOutOfDate = true;
		}
		NetworkData& getNetworkData()
		{
			return m_networkData;
		}
		
		bool m_networkStructureOutOfDate = true;
		NetworkData m_networkData;
		std::vector<ConnectionInfo> m_buildingConnections;
		const std::vector<Neuron::ID> m_inputNeuronIDs;
		const std::vector<Neuron::ID> m_outputNeuronIDs;

		std::unordered_map<Neuron::ID, float> m_biasList;
		std::unordered_map<Neuron::ID, Activation::Type> m_activationTypes;
		std::unordered_map<unsigned int, Activation::Type> m_defaultLayerActivationTypes;
		Activation::Type m_defaultActivationType;

		//bool m_networkBuilt = false;
		std::vector<float> m_inputValues;
		std::vector<float> m_outputValues;
		bool m_enableSoftMax = false;
		float m_softMaxSum = 0;

		std::vector<Visualisation::CustomConnectedNeuralNetPainter*> m_painters;
		
		//LearnAlgo::Backpropagation m_backProp;

		class NEURAL_NET_API CustomConnectedNeuralNetBuilder
		{
		public:
			static void buildNetwork(
				const std::vector<ConnectionInfo>& connections, 
				const std::unordered_map<Neuron::ID, float>& biasList,
				const std::unordered_map<Neuron::ID, Activation::Type>& activationFunctions,
				const std::unordered_map<unsigned int, Activation::Type> &defaultLayerActivationTypes,
				Activation::Type defaultActivationType,
				const std::vector<Neuron::ID> &inputNeuronIDs,
				const std::vector<Neuron::ID> &outputNeuronIDs,
				NetworkData &network);

			static void destroyNetwork(NetworkData& network);

			static void getConnections(const NetworkData& network,
				std::vector<ConnectionInfo>& connectionInfos);

			static void getBiasList(const NetworkData& network,
				std::unordered_map<Neuron::ID, float>& biasList);

		private:
			static void splitIntoLayers(NetworkData& network);
			static void removeDuplicateConnections(const std::vector<ConnectionInfo>& connectionsIn,
				std::vector<ConnectionInfo>& connectionsOut);

			static void sortLayers(NetworkData& network);

			//static void splitIntoLayers(const std::vector<ConnectionInfo>& connections, std::vector<std::vector<ConnectionInfo>>& layers);
			//static void BFS(const std::unordered_map<unsigned int, std::vector<unsigned int>>& adjacencyList, std::vector<std::vector<ConnectionInfo>>& layers);
		};

		
	};	
}
