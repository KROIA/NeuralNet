#pragma once

#include "NeuralNet_base.h"
#include "QSFML_EditorWidget.h"
#include "SimpleImpl/NetworkComponents/Neuron.h"
#include "Visualisation/NeuronPainter.h"

namespace NeuralNet
{
	class CustomConnectedNeuralNet;
	namespace Visualisation
	{
		class NEURAL_NET_EXPORT CustomConnectedNeuralNetPainter : public QSFML::Components::Drawable
		{
			friend class CustomConnectedNeuralNet;
			CustomConnectedNeuralNetPainter(
				CustomConnectedNeuralNet* net,
				const std::string& name = "CustomConnectedNeuralNetPainter");
		public:
			~CustomConnectedNeuralNetPainter();

			void setLayerSpacing(float spacing) { m_layerSpacing = spacing; }
			void setNeuronSpacing(float spacing) { m_neuronSpacing = spacing; }
			void setNeuronRadius(float radius);
			void setNeuronSaturation(float saturation);
			void setNeuronOutlineThickness(float thickness);
			void setNeuronTextSize(float size);
			void setConnectionWidth(float width) { m_connectionWidth = width; }



			void buildNetwork();
			void destroyNetwork();
			void resetPositions();
			void resetLayerPosition(unsigned int layer, const sf::Vector2f &position, const sf::Vector2f &spacing); // start pos of the first neuron

			Neuron::ID getNeuronAtPosition(const sf::Vector2f& position, bool &success) const;
			void setNeuronPosition(Neuron::ID id, const sf::Vector2f& position);
			void moveLayer(unsigned int layer, const sf::Vector2f& offset);
			void moveNeuron(Neuron::ID id, const sf::Vector2f& offset);


			// Enable/Disable the graph of the neuron
			bool isNeuronGraphEnabled(Neuron::ID id) const
			{
				auto it = m_neuronPainters.find(id);
				if (it == m_neuronPainters.end())
					return false;
				return it->second.painter->isGraphEnabled();
			}
			void enableNeuronGraph(Neuron::ID id, bool enable);
			void enableNeuronGraph(bool enable);
			void enableNeuronGraphOfLayer(unsigned int layer, bool enable);

			// Enable/Disable the text of the neuron
			bool isNeuronTextEnabled(Neuron::ID id) const
			{
				auto it = m_neuronPainters.find(id);
				if (it == m_neuronPainters.end())
					return false;
				return it->second.painter->isTextEnabled();
			}
			void enableNeuronText(Neuron::ID id, bool enable);
			void enableNeuronText(bool enable);
			void enableNeuronTextOfLayer(unsigned int layer, bool enable);
		private:
			void setSceneParent(QSFML::Scene* parent) override
			{
				Drawable::setSceneParent(parent);
				//for (auto& pair : m_neuronPainters)
				//{
				//	pair.second.painter->setSceneParent(parent);
				//}
			}
			void drawComponent(sf::RenderTarget& target, sf::RenderStates states) const override;



			float m_signalSatturation = 0.5;
			float m_connectionWidth = 2;
			float m_layerSpacing = 50;
			float m_neuronSpacing = 30;
			float m_neuronRadius = 10;

			CustomConnectedNeuralNet* m_neuralNet;

			struct NeuronPainterData
			{
				sf::Vector2f position;
				Neuron::ID id = 0;
				NeuronPainter* painter = nullptr;
			};

			std::unordered_map<Neuron::ID, NeuronPainterData> m_neuronPainters;

		};
	}
}