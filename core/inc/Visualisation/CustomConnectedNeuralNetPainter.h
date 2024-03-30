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

			void buildNetwork();
			void destroyNetwork();

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