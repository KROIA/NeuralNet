#include "Visualisation/NeuronPainter.h"
#include "Visualisation/Utilities.h"

namespace NeuralNet
{
	namespace Visualisation
	{
		NeuronPainter::NeuronPainter(Neuron* neuron)
			: m_neuron(neuron)
		{
			m_idText.setOrigin(QSFML::Utilities::Origin::Center);
			m_idText.setText(std::to_string(neuron->getID()));
			
			m_idText.setScale(0.1);
			m_idText.setPosition(sf::Vector2f(0,-m_radius*0.8));
		}

		NeuronPainter::~NeuronPainter()
		{
			m_neuron->removePainter(this);
		}

		void NeuronPainter::drawComponent(sf::RenderTarget& target, sf::RenderStates states) const
		{
			sf::CircleShape shape(m_radius);
			float signalValue = m_neuron->getOutput();
			shape.setOrigin(m_radius, m_radius);
			shape.setFillColor(Utilities::signalColor(signalValue * m_saturation));
			shape.setOutlineColor(sf::Color::Black);
			shape.setOutlineThickness(1);
			target.draw(shape, states);

			//std::vector<sf::Vertex> points(20);
			//getFunctionGraph(points, sf::Vector2f(m_radius, m_radius*0.5));

			//states.transform.translate(sf::Vector2f(m_radius*0.4, 0));
			//target.draw(points.data(), points.size(), sf::LineStrip, states);

			drawFunctionGraph(target, states);

			m_idText.draw(target, states);
		}

		/*void NeuronPainter::getFunctionGraph(std::vector<sf::Vertex>& points, const sf::Vector2f scale) const
		{
			size_t count = points.size();
			float startX = -5;
			float xScale = scale.x / (2*std::abs(startX));
			float step = (-2* startX) / (count - 1);
			
			Activation::ActivationFunction function = Activation::getActivationFunction(m_neuron->getActivationType());
			
			float x = startX;
			float y = function(x);
			
			for (size_t i = 0; i < count; ++i)
			{
				x = i * step + startX;
				y = function(x);
				points[i].position = sf::Vector2f(x * xScale, -y * scale.y);
			}
		}*/

		void NeuronPainter::drawFunctionGraph(sf::RenderTarget& target, sf::RenderStates states) const
		{
			sf::Vector2f frameSize = sf::Vector2f(m_radius*2, m_radius) * 0.6f;
			states.transform.translate(-frameSize.x*0.5f, 0);
			sf::Color crossColor = sf::Color::Blue;
			float crossSize = 1;
			float startX = -5;
			float endX = -startX;

			// Add grid to background of the graph
			sf::VertexArray grid(sf::Lines, 2 * 2);
			grid[0].position = sf::Vector2f(0, 0);
			grid[0].color = sf::Color::Black;
			grid[1].position = sf::Vector2f(frameSize.x, 0);
			grid[1].color = sf::Color::Black;

			grid[2].position = sf::Vector2f(frameSize.x * 0.5, -frameSize.y);
			grid[2].color = sf::Color::Black;
			grid[3].position = sf::Vector2f(frameSize.x * 0.5, frameSize.y);
			grid[3].color = sf::Color::Black;
			target.draw(grid, states);

			// Draw graph of activation function
			std::vector<float> points(50);
			
			float dx = (endX - startX) / (points.size() - 1);

			Activation::ActivationFunction function = Activation::getActivationFunction(m_neuron->getActivationType());
			float minY = -1;
			float maxY = 1;
			for (size_t i = 0; i < points.size(); ++i)
			{
				float x = startX + i * dx;
				float y = function(x);
				points[i] = y;
				if(y < minY)
					minY = y;
				if(y > maxY)
					maxY = y;
			}
			float maxAbs = std::max(std::abs(minY), std::abs(maxY));
			sf::VertexArray lines(sf::LinesStrip, points.size());
			float xPosFactor = frameSize.x / (points.size() - 1);

			float yScale = -(frameSize.y) / (maxAbs);

			for (size_t i = 0; i < points.size(); ++i)
			{
				lines[i].position = sf::Vector2f(i * xPosFactor, points[i] * yScale);
				lines[i].color = sf::Color::White;
			}
			target.draw(lines, states);

			// Draw current activation as cross
			sf::VertexArray cross(sf::Lines, 4);
			float x = m_neuron->getNetInput();
			float y = function(x);

			float xPos = (x - startX) / (endX - startX) * frameSize.x;
			if(xPos >= frameSize.x)
				xPos = frameSize.x;
			else if(xPos < -frameSize.x)
				xPos = -frameSize.x;
			
			cross[0].position = sf::Vector2f(xPos - crossSize, yScale * y - crossSize);
			cross[1].position = sf::Vector2f(xPos + crossSize, yScale * y + crossSize);
															   
			cross[2].position = sf::Vector2f(xPos + crossSize, yScale * y - crossSize);
			cross[3].position = sf::Vector2f(xPos - crossSize, yScale * y + crossSize);

			
			cross[0].color = crossColor;
			cross[1].color = crossColor;
			cross[2].color = crossColor;
			cross[3].color = crossColor;
			target.draw(cross, states);


		}
	}
}