#pragma once

#include <QMainWindow>
#include "QSFML_EditorWidget.h"
#include "NeuralNet.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget* parent = nullptr);
    ~MainWindow();

private slots:
    void onTimerFinish();
private:
    void setupCanvas();
    void closeEvent(QCloseEvent* event) override;


    Ui::MainWindow* ui;

    QSFML::Scene* m_scene;

    //NeuralNet::Visualisation::VisuFullConnectedNeuronalNet* m_visuNet;
    NeuralNet::FullConnectedNeuralNet* m_net;
    NeuralNet::CustomConnectedNeuralNet * m_customNet;

    NeuralNet::NeuralNetCanvasObject *m_netObject1;
    NeuralNet::NeuralNetCanvasObject *m_netObject2;

    QTimer m_timer;

    struct TrainingSample
    {
        std::vector<float> inputs;
        std::vector<float> expectedOutput;
    };

    std::vector<TrainingSample> m_trainingData;
	int m_currentExampleIndex = 0;
};
