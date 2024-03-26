#pragma once

#include "NeuralNet_base.h"
#include "Base/NeuralNetBase.h"
#include "SimpleImpl/NetworkComponents/Neuron.h"
#include "SimpleImpl/NetworkComponents/Connection.h"
#include "SimpleImpl/NetworkComponents/Layer.h"
#include "QSFML_EditorWidget.h"
#include <unordered_map>

namespace NeuralNet
{
	class NEURAL_NET_EXPORT CustomConnectedNeuralNet : public NeuralNetBase
	{
		friend class CustomConnectedNeuralNetPainter;
	public:
		struct ConnectionInfo
		{
			unsigned int fromNeuronID;
			unsigned int toNeuronID;
			float weight;
		};
		using NeuronID = unsigned int;
		class NEURAL_NET_EXPORT CustomConnectedNeuralNetPainter : public QSFML::Components::Drawable
		{
			friend class CustomConnectedNeuralNet;
			CustomConnectedNeuralNetPainter(
				CustomConnectedNeuralNet *net, 
				const std::string& name = "CustomConnectedNeuralNetPainter");

			~CustomConnectedNeuralNetPainter();
		public:


		private:
			void drawComponent(sf::RenderTarget& target, sf::RenderStates states) const override;

			sf::Color signalColor(float value) const
			{
				value = m_signalSatturation * value;
				return QSFML::Color::lerpLinear({ m_lowValueColor, m_mediumValueColor, m_highValueColor }, (value + 1.f) / 2.f);
			}
			float signalWidth(float weight) const
			{
				weight = sqrt(std::abs(weight * m_connectionWidth));
				if (weight < 1)
					return 1;
				return weight;
			}

			sf::Color m_lowValueColor = sf::Color::Red;
			sf::Color m_mediumValueColor = sf::Color(150, 150, 150);
			sf::Color m_highValueColor = sf::Color::Green;
			float m_signalSatturation = 0.5;
			float m_connectionWidth = 2;
			CustomConnectedNeuralNet* m_neuralNet;
		};


		CustomConnectedNeuralNet(
			unsigned int inputSize,
			unsigned int outputSize);

		~CustomConnectedNeuralNet();

		
		void addConnection(const ConnectionInfo& connectionInfo);
		void addConnection(NeuronID fromNeuronID, NeuronID toNeuronID, float weight);
		void setConnections(const std::vector<ConnectionInfo>& connections);

		void buildNetwork();
		void destroyNetwork();

		void setInputValues(const std::vector<float>& values) override;
		void setInputValue(unsigned int index, float values) override;
		std::vector<float> getOutputValues() const override;
		float getOutputValue(unsigned int index) const override;

		void update() override;

		CustomConnectedNeuralNetPainter* createVisualisation();

	private:
		void removePainter(CustomConnectedNeuralNetPainter* painter);
		struct NetworkData
		{
			/// <summary>
			/// ID - Instance pair
			/// </summary>
			std::unordered_map<NeuronID, Neuron*> neurons;

			/// <summary>
			/// Container for all connections
			/// </summary>
			std::vector<Connection*> connections;

			/// <summary>
			/// Same objects, but splitted into layers
			/// </summary>
			std::vector<Layer> layers;
		};

		NetworkData m_networkData;
		std::vector<ConnectionInfo> m_buildingConnections;
		bool m_networkBuilt = false;
		std::vector<float> m_inputValues;
		std::vector<float> m_outputValues;

		std::vector<CustomConnectedNeuralNetPainter*> m_painters;
		

		class NEURAL_NET_EXPORT CustomConnectedNeuralNetBuilder
		{
		public:
			static void buildNetwork(
				const std::vector<ConnectionInfo>& connections, 
				unsigned int inputCount, 
				unsigned int outputCount,
				NetworkData &network);

			static void destroyNetwork(NetworkData& network);

			static void getConnections(const NetworkData& network,
				std::vector<ConnectionInfo>& connectionInfos);

		private:
			static void splitIntoLayers(NetworkData& network);

			//static void splitIntoLayers(const std::vector<ConnectionInfo>& connections, std::vector<std::vector<ConnectionInfo>>& layers);
			//static void BFS(const std::unordered_map<unsigned int, std::vector<unsigned int>>& adjacencyList, std::vector<std::vector<ConnectionInfo>>& layers);
		};

		
	};	
}
