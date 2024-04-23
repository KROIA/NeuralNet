#pragma once

#include "NeuralNet_base.h"
#include "QSFML_EditorWidget.h"
#include <QObject>
#include "CustomConnectedNeuralNetPainter.h"
#include "SimpleImpl/Nets/CustomConnectedNeuralNet.h"

//#include <qspinbox.h>

namespace NeuralNet
{
	class NEURAL_NET_EXPORT NeuralNetCanvasObject: public QObject, public QSFML::Objects::CanvasObject
	{
		Q_OBJECT
	public:
		OBJECT_DECL(NeuralNetCanvasObject);

		NeuralNetCanvasObject(CustomConnectedNeuralNet *net,
							  const std::string& name = "NeuralNetCanvasObject",
							  CanvasObject* parent = nullptr);
		NeuralNetCanvasObject(const NeuralNetCanvasObject& other);
		~NeuralNetCanvasObject();

		void setLayerSpacing(float spacing) { m_neuralNetPainter->setLayerSpacing(spacing); }
		void setNeuronSpacing(float spacing) { m_neuralNetPainter->setNeuronSpacing(spacing); }
		void buildNetwork() { m_neuralNetPainter->buildNetwork(); }
		void resetPositions() { m_neuralNetPainter->resetPositions(); }
		void setNeuronRadius(float radius) { m_neuralNetPainter->setNeuronRadius(radius); }

		void setNeuronPosition(Neuron::ID id, const sf::Vector2f& pos) { m_neuralNetPainter->setNeuronPosition(id, pos); }
		void moveLayer(unsigned int layer, const sf::Vector2f& offset) { m_neuralNetPainter->moveLayer(layer, offset); }
		void moveNeuron(Neuron::ID id, const sf::Vector2f& offset) { m_neuralNetPainter->moveNeuron(id, offset); }
		void resetLayerPosition(unsigned int layer, const sf::Vector2f& position, const sf::Vector2f& spacing)
		{ m_neuralNetPainter->resetLayerPosition(layer, position, spacing); }
		


		void update() override;

		void setDragingGridSize(int gridSize)
		{
			m_dragData.gridSize = gridSize;
		}
		int getDragingGridSize() const
		{
			return m_dragData.gridSize;
		}

	private slots:
		void onMouseFallingEdge();
		void onMouseRisingEdge();

	protected:

		

	private:
		void setup();


		QSFML::Components::MousePressEvent *m_mousePressEvent;
		Visualisation::CustomConnectedNeuralNetPainter *m_neuralNetPainter;
		CustomConnectedNeuralNet *m_neuralNet;

		struct DragData
		{
			int gridSize = 10;
			Neuron* dragingNeuron = nullptr;
		};
		DragData m_dragData;

		//QSpinBox* m_spinBox = nullptr;

	};
}