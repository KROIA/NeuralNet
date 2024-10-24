#include "Visualisation/NeuralNetCanvasObject.h"
#include <QLayout>
#include <QWindow>
#include <iostream>

namespace NeuralNet
{

	OBJECT_IMPL(NeuralNetCanvasObject);


	NeuralNetCanvasObject::NeuralNetCanvasObject(
		CustomConnectedNeuralNet* net,
		const std::string& name,
		GameObject* parent)
		: GameObject(name, parent)
		, m_neuralNet(net)
	{
		setup();
	}
	NeuralNetCanvasObject::NeuralNetCanvasObject(const NeuralNetCanvasObject& other)
		: GameObject(other)
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
			sf::Vector2i pixelPos = getMousePosition();
			//sf::Vector2i widgetSize(getCanvasParent()->geometry().width(), 
			//	getCanvasParent()->geometry().height());
			/*if (m_spinBox)
			{
				//std::cout << "ratio: "<< getCanvasParent()->devicePixelRatio();
				auto geometry = m_spinBox->geometry();
				geometry.moveTo(pixelPos.x, pixelPos.y);
				m_spinBox->setGeometry(geometry);
				//std::cout << "Spinbox pos: " << pixelPos.x << " " << pixelPos.y;
				//std::cout << "WidgetSize: " << widgetSize.x << " " << widgetSize.y << std::endl;
			}*/
			
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
		//m_spinBox = new QSpinBox(getCanvasParent());
		//m_spinBox->show();
		
		//getCanvasParent()->layout()->addWidget(m_spinBox);
	}
	void NeuralNetCanvasObject::onMouseRisingEdge()
	{
		m_dragData.dragingNeuron = nullptr;
		//delete m_spinBox;
		//m_spinBox = nullptr;
	}
}