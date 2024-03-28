#pragma once

#include "NeuralNet_base.h"
#include "QSFML_EditorWidget.h"
#include "SimpleImpl/NetworkComponents/Neuron.h"

namespace NeuralNet
{
	namespace Visualisation
	{
		class NEURAL_NET_EXPORT NeuronPainter : public QSFML::Components::Drawable
		{
			friend Neuron;
			NeuronPainter(Neuron* neuron);
		public:
			~NeuronPainter();

			Neuron* getNeuron() const
			{
				return m_neuron;
			}

			sf::Vector2f getOutputConnectionPoint() const
			{
				return getPosition();// +sf::Vector2f(m_radius, 0);
			}
			sf::Vector2f getInputConnectionPoint() const
			{
				return getPosition();// -sf::Vector2f(m_radius, 0);
			}

			void setCanvasParent(QSFML::Canvas* parent) override
			{
				Drawable::setCanvasParent(parent);
				m_idText.setCanvasParent(parent);
				if(parent)
					m_idText.setFont(parent->getTextFont());
			}
		protected:

			

		private:

			void drawComponent(sf::RenderTarget& target, sf::RenderStates states) const override;


			//void getFunctionGraph(std::vector<sf::Vertex>& points, const sf::Vector2f scale) const;
			void drawFunctionGraph(sf::RenderTarget& target, sf::RenderStates states) const;

			Neuron *m_neuron;

			QSFML::Components::Text m_idText;

			float m_radius = 10;
			float m_saturation = 0.5;
		};
	}
}