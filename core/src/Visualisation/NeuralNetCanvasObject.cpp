#include "Visualisation/NeuralNetCanvasObject.h"

namespace NeuralNet
{

	OBJECT_IMPL(NeuralNetCanvasObject);


	NeuralNetCanvasObject::NeuralNetCanvasObject(
		CustomConnectedNeuralNet* net,
		const std::string& name,
		CanvasObject* parent)
		: CanvasObject(name, parent)
		, m_neuralNet(net)
	{
		setup();
	}
	NeuralNetCanvasObject::NeuralNetCanvasObject(const NeuralNetCanvasObject& other)
		: CanvasObject(other)
		, m_neuralNet(other.m_neuralNet)
	{
		setup();
	}
	NeuralNetCanvasObject::~NeuralNetCanvasObject()
	{

	}

	void NeuralNetCanvasObject::setup()
	{
		m_mousePressEvent = new QSFML::Components::MousePressEvent();
		m_mousePressEvent->setTriggerButton(sf::Mouse::Button::Right);
		connect(m_mousePressEvent, &QSFML::Components::MousePressEvent::fallingEdge,
				this, &NeuralNetCanvasObject::onMouseFallingEdge);
		connect(m_mousePressEvent, &QSFML::Components::MousePressEvent::risingEdge,
			this, &NeuralNetCanvasObject::onMouseRisingEdge);

		addComponent(m_mousePressEvent);


		m_neuralNetPainter = m_neuralNet->createVisualisation();
		addComponent(m_neuralNetPainter);
	}


	void NeuralNetCanvasObject::update()
	{
		if (m_dragData.dragingNeuron && m_neuralNetPainter)
		{
			sf::Vector2f newPos = getMouseWorldPosition() - m_neuralNetPainter->getGlobalPosition();
			if (sf::Keyboard::isKeyPressed(sf::Keyboard::LShift))
			{
				// snap to grid
				int gridsize = m_dragData.gridSize;
				newPos.x = std::round(newPos.x / gridsize) * gridsize;
				newPos.y = std::round(newPos.y / gridsize) * gridsize;
			}
			m_neuralNetPainter->setNeuronPosition(m_dragData.dragingNeuron->getID(), newPos);
		}
	}


	void NeuralNetCanvasObject::onMouseFallingEdge()
	{
		if (!m_neuralNetPainter)
			return;
		sf::Vector2f relativeWorldPos = m_mousePressEvent->getLastPressedWorldPos() - m_neuralNetPainter->getGlobalPosition();
		bool success = false;
		Neuron::ID selectedNeuron = m_neuralNetPainter->getNeuronAtPosition(relativeWorldPos, success);
		if(!success)
			return;

		std::cout << "Selected neuron: " << selectedNeuron << std::endl;
		m_dragData.dragingNeuron = m_neuralNet->getNeuron(selectedNeuron);
	}
	void NeuralNetCanvasObject::onMouseRisingEdge()
	{
		m_dragData.dragingNeuron = nullptr;
	}
}