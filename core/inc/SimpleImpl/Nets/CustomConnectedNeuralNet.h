#pragma once

#include "NeuralNet_base.h"
#include "Base/NeuralNetBase.h"
#include "SimpleImpl/Nets/NetworkData.h"
#include "SimpleImpl/NetworkComponents/Neuron.h"
#include "SimpleImpl/NetworkComponents/Connection.h"
#include "SimpleImpl/NetworkComponents/Layer.h"
#include "QSFML_EditorWidget.h"
#include <unordered_map>

#include "LearnAlgo/Backpropagation.h"

namespace NeuralNet
{
	namespace Visualisation
	{
		class CustomConnectedNeuralNetPainter;
	}
	class NEURAL_NET_EXPORT CustomConnectedNeuralNet : public NeuralNetBase
	{
		friend class Visualisation::CustomConnectedNeuralNetPainter;
	public:
		CustomConnectedNeuralNet(
			unsigned int inputSize,
			unsigned int outputSize);

		~CustomConnectedNeuralNet();

		
		void clearConnections()
		{
			m_buildingConnections.clear();
			m_activationFunctions.clear();
		}
		void addConnection(const ConnectionInfo& connectionInfo);
		void addConnection(Neuron::ID fromNeuronID, Neuron::ID toNeuronID);
		void addConnection(Neuron::ID fromNeuronID, Neuron::ID toNeuronID, float weight);
		void setConnections(const std::vector<ConnectionInfo>& connections);
		void removeConnection(Neuron::ID fromNeuronID, Neuron::ID toNeuronID);
		std::vector<ConnectionInfo> getConnections() const;

		void buildNetwork();
		void destroyNetwork();

		void setInputValues(const std::vector<float>& values) override;
		void setInputValue(unsigned int index, float values) override;
		std::vector<float> getOutputValues() const override;
		float getOutputValue(unsigned int index) const override;

		Activation::Type getActivationType(Neuron::ID id) const;
		void setActivationType(Neuron::ID id, Activation::Type type);
		void setLayerActivationType(unsigned int layerIdx, Activation::Type type);
		void setActivationType(Activation::Type type);

		Neuron* getNeuron(Neuron::ID id);

		void update() override;

		Visualisation::CustomConnectedNeuralNetPainter* createVisualisation();

		void learn(const std::vector<float>& expectedOutput);
		std::vector<float> getOutputError(const std::vector<float>& expectedOutput) const;
		float getNetError(const std::vector<float>& expectedOutput) const;

	private:
		void removePainter(Visualisation::CustomConnectedNeuralNetPainter* painter);
		

		NetworkData m_networkData;
		std::vector<ConnectionInfo> m_buildingConnections;

		std::unordered_map<Neuron::ID, Activation::Type> m_activationFunctions;
		std::unordered_map<unsigned int, Activation::Type> m_defaultLayerActivationTypes;
		Activation::Type m_defaultActivationType;

		bool m_networkBuilt = false;
		std::vector<float> m_inputValues;
		std::vector<float> m_outputValues;

		std::vector<Visualisation::CustomConnectedNeuralNetPainter*> m_painters;
		
		LearnAlgo::Backpropagation m_backProp;

		class NEURAL_NET_EXPORT CustomConnectedNeuralNetBuilder
		{
		public:
			static void buildNetwork(
				const std::vector<ConnectionInfo>& connections, 
				const std::unordered_map<Neuron::ID, Activation::Type>& activationFunctions,
				const std::unordered_map<unsigned int, Activation::Type> &defaultLayerActivationTypes,
				Activation::Type defaultActivationType,
				unsigned int inputCount, 
				unsigned int outputCount,
				NetworkData &network);

			static void destroyNetwork(NetworkData& network);

			static void getConnections(const NetworkData& network,
				std::vector<ConnectionInfo>& connectionInfos);

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
