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
		
		// Enable/Disable the graph of the neuron
		bool isNeuronGraphEnabled(Neuron::ID id) const { return m_neuralNetPainter->isNeuronGraphEnabled(id); }
		void enableNeuronGraph(Neuron::ID id, bool enable){ m_neuralNetPainter->enableNeuronGraph(id, enable); }
		void enableNeuronGraph(bool enable){ m_neuralNetPainter->enableNeuronGraph(enable); }
		void enableNeuronGraphOfLayer(unsigned int layer, bool enable){ m_neuralNetPainter->enableNeuronGraphOfLayer(layer, enable); }

		// Enable/Disable the text of the neuron
		bool isNeuronTextEnabled(Neuron::ID id) const { return m_neuralNetPainter->isNeuronTextEnabled(id); }
		void enableNeuronText(Neuron::ID id, bool enable){ m_neuralNetPainter->enableNeuronText(id, enable); }
		void enableNeuronText(bool enable){ m_neuralNetPainter->enableNeuronText(enable); }
		void enableNeuronTextOfLayer(unsigned int layer, bool enable){ m_neuralNetPainter->enableNeuronTextOfLayer(layer, enable); }

		void setDragingGridSize(int gridSize)
		{
			m_dragData.gridSize = gridSize;
		}
		int getDragingGridSize() const
		{
			return m_dragData.gridSize;
		}

		CustomConnectedNeuralNet* getNeuralNet() const
		{
			return m_neuralNet;
		}
		unsigned int getInputLayerIndex() const
		{
			return m_neuralNet->getInputLayerIndex();
		}
		unsigned int getOutputLayerIndex() const
		{
			return m_neuralNet->getOutputLayerIndex();
		}

		void update() override;

		



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