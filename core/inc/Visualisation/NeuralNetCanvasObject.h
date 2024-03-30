#pragma once

#include "NeuralNet_base.h"
#include "QSFML_EditorWidget.h"
#include <QObject>
#include "CustomConnectedNeuralNetPainter.h"
#include "SimpleImpl/Nets/CustomConnectedNeuralNet.h"

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

		Neuron *m_currentSelectedNeuron = nullptr;
	};
}