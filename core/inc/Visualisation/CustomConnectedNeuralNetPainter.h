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


			void buildNetwork();
			void destroyNetwork();
			void resetPositions();
			void resetLayerPosition(unsigned int layer, const sf::Vector2f &position, const sf::Vector2f &spacing); // start pos of the first neuron

			Neuron::ID getNeuronAtPosition(const sf::Vector2f& position, bool &success) const;
			void setNeuronPosition(Neuron::ID id, const sf::Vector2f& position)
			{
				auto it = m_neuronPainters.find(id);
				if (it != m_neuronPainters.end())
				{
					it->second.position = position;
					it->second.painter->setPosition(position);
				}
			}
			void moveLayer(unsigned int layer, const sf::Vector2f& offset);
			void moveNeuron(Neuron::ID id, const sf::Vector2f& offset)
			{
				auto it = m_neuronPainters.find(id);
				if (it != m_neuronPainters.end())
				{
					it->second.position += offset;
					it->second.painter->move(offset);
				}
			}
		private:
			void setCanvasParent(QSFML::Canvas* parent) override
			{
				Drawable::setCanvasParent(parent);
				for (auto& pair : m_neuronPainters)
				{
					pair.second.painter->setCanvasParent(parent);
				}
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
				Neuron::ID id;
				NeuronPainter* painter;
			};

			std::unordered_map<Neuron::ID, NeuronPainterData> m_neuronPainters;

		};
	}
}