#include "LearnAlgo/GeneticLearn.h"

namespace NeuralNet
{
	namespace LearnAlgo
	{
		float GeneticLearn::m_mutationRate = 0.1f;
		int GeneticLearn::m_mutationCountPercentage = 10;
		int GeneticLearn::m_crossoverCountPercentage = 50;

		Log::LogObject& GeneticLearn::getLogger()
		{
			static Log::LogObject logger("GeneticLearn");
			return logger;
		}
		void GeneticLearn::learnAndReplace(std::vector<GeneticPerformance>& nets)
		{
			std::vector<std::vector<float>> genom;
			learn(nets, genom);
			for (size_t i = 0; i < nets.size(); ++i)
			{
				nets[i].net->setGenom(genom[i]);
			} 
		}
		void GeneticLearn::learn(
			std::vector<GeneticPerformance>& nets,
			std::vector<std::vector<float>>& genom)
		{
			genom.resize(nets.size());
			for (size_t i = 0; i < nets.size(); ++i)
			{
				genom[i] = nets[i].net->getGenom();
			}

			std::vector<FitnessIndex> sortedIndices = getSortedFitnessIndices(nets);
			float sumFitness = getSumFitness(nets);

			for (size_t i = 0; i < nets.size()/2; ++i)
			{
				size_t index1 = select(sortedIndices, sumFitness);
				size_t index2 = select(sortedIndices, sumFitness);
				size_t tryCount = 0;
				while (index1 == index2 && tryCount < 10)
				{
					++tryCount;
					index2 = select(sortedIndices, sumFitness);
				}

				if (tryCount >= 10)
				{
					getLogger().logWarning("learn(...): Could not find different indices for crossover");
					//continue;
				}
				crossover(genom[index1], genom[index2], m_crossoverCountPercentage);
				mutate(genom[index1], m_mutationCountPercentage);
				mutate(genom[index2], m_mutationCountPercentage);
			}
		}



		std::vector<GeneticLearn::FitnessIndex> GeneticLearn::getSortedFitnessIndices(const std::vector<GeneticPerformance>& nets)
		{
			std::vector<FitnessIndex> indices(nets.size());
			for (size_t i = 0; i < nets.size(); ++i)
			{
				indices[i].index = i;
				indices[i].fitness = nets[i].fitness;
			}
			
			std::sort(indices.begin(), indices.end(), [&nets](FitnessIndex &i1, FitnessIndex &i2)
				{return i1.fitness < i2.fitness; });
			return indices;
		}
		float GeneticLearn::getSumFitness(const std::vector<GeneticPerformance>& nets)
		{
			float sum = 0;
			for (size_t i=0; i< nets.size(); ++i)
			{
				sum += nets[i].fitness;
			}
			return sum;
		}
		size_t GeneticLearn::select(const std::vector<FitnessIndex>& sortedIndices, float sumFitness)
		{
			float randomValue = (rand() % 1000) * sumFitness / 1000.0f;
			float sum = 0;
			for (size_t i = 0; i < sortedIndices.size(); ++i)
			{
				float nextSum = sum + sortedIndices[i].fitness;
				if (randomValue >= sum && randomValue < nextSum)
				{
					return sortedIndices[i].index;
				}
				sum = nextSum;
			}
			getLogger().logWarning("select(...): No index selected, strange...");
			return sortedIndices.back().index;
		}
		void GeneticLearn::mutate(std::vector<float>& genom, int mutationCountPercentage)
		{
			for (size_t i = 0; i < genom.size(); ++i)
			{
				if (rand() % 100 < mutationCountPercentage)
				{
					float mutation = ((rand() % 200)-100) * m_mutationRate;
					genom[i] += mutation;
				}
			}
		}
		void GeneticLearn::crossover(std::vector<float>& genom1, std::vector<float>& genom2, int crossoverCountPercentage)
		{
			size_t genomSize = genom1.size();
			if (genomSize != genom2.size()) [[unlikely]]
			{
				getLogger().logError("crossover(...): Genom size mismatch");
				return;
			}

			for (size_t i = 0; i < genomSize; ++i)
			{
				if (rand() % 100 < crossoverCountPercentage)
				{
					std::swap(genom1[i], genom2[i]);
				}
			}
		}

	}
}