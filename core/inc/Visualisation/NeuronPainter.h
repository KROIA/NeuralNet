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
				return getPosition() -sf::Vector2f(m_radius, 0);
			}
			void setRadius(float radius)
			{
				m_radius = radius;
			}
			float getRadius() const
			{
				return m_radius;
			}

			void setCanvasParent(QSFML::Canvas* parent) override
			{
				Drawable::setCanvasParent(parent);
				m_idText.setCanvasParent(parent);
				if(parent)
					m_idText.setFont(parent->getTextFont());
			}

			bool contains(const sf::Vector2f& point) const
			{
				const sf::Vector2f &pos = getPosition();
				float distance2 = (pos.x - point.x) * (pos.x - point.x) + (pos.y - point.y) * (pos.y - point.y);
				return  distance2  <= m_radius * m_radius;
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
			float m_outlineThickness = 1;

			bool m_enableGraph = true;
		};
	}
}