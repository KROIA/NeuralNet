#pragma once

#include <QMainWindow>
#include "QSFML_EditorWidget.h"
#include "NeuralNet.h"
#include "Dataset.h"

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

    void on_pauseTraining_pushButton_clicked();
    void on_testNext_pushButton_clicked();
private:
    void setupCanvas();
    void closeEvent(QCloseEvent* event) override;
    void train(size_t iterations);
    float test(const Dataset::DataPoint& dataPoint);


    Ui::MainWindow* ui;

    QSFML::Canvas* m_canvas;

    //NeuralNet::Visualisation::VisuFullConnectedNeuronalNet* m_visuNet;
    NeuralNet::FullConnectedNeuralNet* m_net;
    NeuralNet::CustomConnectedNeuralNet * m_customNet;

    NeuralNet::NeuralNetCanvasObject *m_netObject1;
    NeuralNet::NeuralNetCanvasObject *m_netObject2;

    QTimer m_timer;

    Dataset m_dataset;
    Dataset m_validationSet;
};
