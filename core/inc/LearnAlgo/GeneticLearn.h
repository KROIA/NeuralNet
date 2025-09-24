#pragma once

#include "NeuralNet_base.h"
#include <vector>
#include "SimpleImpl/NetworkComponents/Layer.h"
#include "SimpleImpl/Nets/FullConnectedNeuralNet.h"
#include "SimpleImpl/Nets/CustomConnectedNeuralNet.h"

namespace NeuralNet
{
	namespace LearnAlgo
	{
		class NEURAL_NET_API GeneticLearn
		{
			GeneticLearn() = delete;
		public:
			static Log::LogObject& getLogger();
			struct GeneticPerformance
			{
				CustomConnectedNeuralNet* net;
				float fitness;
			};

			static void setMutationRate(float mutationRate) { m_mutationRate = mutationRate; }
			static float getMutationRate() { return m_mutationRate; }
			static void setMutationCountPercentage(int mutationCountPercentage) { m_mutationCountPercentage = mutationCountPercentage; }
			static int getMutationCountPercentage() { return m_mutationCountPercentage; }
			static void setCrossoverCountPercentage(int crossoverCountPercentage) { m_crossoverCountPercentage = crossoverCountPercentage; }
			static int getCrossoverCountPercentage() { return m_crossoverCountPercentage; }
			
			static void learnAndReplace(std::vector<GeneticPerformance>& nets);
			static void learn(std::vector<GeneticPerformance>& nets, std::vector<std::vector<float>> &genom);
		private:
			struct FitnessIndex
			{
				size_t index;
				float fitness;
			};

			static std::vector<FitnessIndex> getSortedFitnessIndices(const std::vector<GeneticPerformance>& nets);
			static float getSumFitness(const std::vector<GeneticPerformance>& nets);
			static size_t select(const std::vector<FitnessIndex>& sortedIndices, float sumFitness);
			static void mutate(std::vector<float>& genom, int mutationCountPercentage);
			static void crossover(std::vector<float>& genom1, std::vector<float>& genom2, int crossoverCountPercentage);

			static float m_mutationRate;			 ///< The rate of mutation deltaWeight = mutationRate * random(-1, 1)
			static int m_mutationCountPercentage;    ///< The amout of mutation in percent of genom size
			static int m_crossoverCountPercentage;   ///< The amout of crossover in percent of genom size
		};
	}
}