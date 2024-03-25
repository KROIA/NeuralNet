#include "MainWindow.h"
#include "ui_MainWindow.h"
#include <iostream>
#include <QCloseEvent>
#include <QDebug>

using namespace QSFML;
using namespace QSFML::Objects;

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    m_canvas = nullptr;
    setupCanvas();
    
    

    
}

MainWindow::~MainWindow()
{
    delete ui;
    delete m_canvas;
}

void MainWindow::setupCanvas()
{
    CanvasSettings settings;
    //settings.layout.autoAjustSize = false;
    settings.layout.fixedSize = sf::Vector2u(300, 100);
    settings.contextSettings.antialiasingLevel = 8;
    settings.timing.frameTime = 0;
    //settings.updateControlls.enableMultithreading = false;
    //settings.updateControlls.enablePaintLoop = false;
    //settings.updateControlls.enableEventLoop = false;
    //settings.updateControlls.enableUpdateLoop = false;
    m_canvas = new Canvas(ui->canvasWidget, settings);

    DefaultEditor* defaultEditor = new DefaultEditor();
    m_canvas->addObject(defaultEditor);
    qDebug() << defaultEditor->toString().c_str();

    m_net = new NeuralNet::FullConnectedNeuralNet(2, 1, 1, 5);
    m_visuNet = new NeuralNet::Visualisation::VisuFullConnectedNeuronalNet(m_net);
    std::vector<float> weights = m_net->getWeights();
    for (size_t i = 0; i < weights.size(); i++)
    {
        weights[i] = (2.f *i / weights.size())-1.f;
	}
    m_net->setWeights(weights);
    m_canvas->addObject(m_visuNet);
}
void MainWindow::closeEvent(QCloseEvent* event)
{
    if (m_canvas)
        m_canvas->stop();
    event->accept();
}

