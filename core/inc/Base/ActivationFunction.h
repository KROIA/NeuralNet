#pragma once

#include "NeuralNet_base.h"


namespace NeuralNet
{
	namespace Activation
	{
		typedef float (*ActivationFunction)(float);
		enum Type
		{
			linear,
			finiteLinear,
			relu,
			binary,
			sigmoid,
			gaussian,
			tanh_,

			count
		};
		inline float activate_linear(float x) { return x; }
		inline float activate_linear_derivetive(float x) { NN_UNUSED(x);  return 1; }
		inline float activate_finiteLinear(float x) { return x < -1 ? -1.f : x>1 ? 1.f : x; }
		inline float activate_finiteLinear_derivetive(float x) { return x < -1 ? 0.f : x>1 ? 0.f : 1.f; }
		inline float activate_relu(float x) { return x <= 0 ? 0.f : x; }
		inline float activate_relu_derivetive(float x) { return x <= 0 ? 0.f : 1.f; }
		inline float activate_binary(float x) { return x < 0 ? 0.f : 1.f; }
		inline float activate_binary_derivetive(float x) { return x == 0 ? 1.f : 0.f; }

		//https://www.wolframalpha.com/input/?i=exp%28-pow%28x%2C2%29%29*2-1
		inline float activate_gaussian(float x) { return exp(-(x) * (x)); }
		inline float activate_gaussian_derivetive(float x) { return -2.f * exp(-(x) * (x)) * x; }
		inline float activate_sigmoid(float x) { return 1 / (1 + exp(-x)); }
		inline float activate_sigmoid_derivetive(float x) {
			float ex = exp(x);
			float sum = ex + 1;
			return ex / (sum * sum);
		}

		inline float activation_tanh(float x) { return tanh(x); }
		inline float activation_tanh_derivetive(float x) {
			float c = cosh(x);
			return 1 / (c * c);
		}

		inline ActivationFunction getActivationFunction(Type type)
		{
			switch (type)
			{
			case Type::linear: return &activate_linear;
			case Type::finiteLinear: return &activate_finiteLinear;
			case Type::relu: return &activate_relu;
			case Type::binary: return &activate_binary;
			case Type::gaussian: return &activate_gaussian;
			case Type::sigmoid: return &activate_sigmoid;
			case Type::tanh_: return &activation_tanh;
			}
			return activate_linear;
		}
		inline ActivationFunction getActivationDerivetiveFunction(Type type)
		{
			switch (type)
			{
			case Type::linear: return &activate_linear_derivetive;
			case Type::finiteLinear: return &activate_finiteLinear_derivetive;
			case Type::relu: return &activate_relu_derivetive;
			case Type::binary: return &activate_binary_derivetive;
			case Type::gaussian: return &activate_gaussian_derivetive;
			case Type::sigmoid: return &activate_sigmoid_derivetive;
			case Type::tanh_: return &activation_tanh_derivetive;
			}
			return nullptr;
		}

		inline const char* getActivationName(Type type)
		{
			switch (type)
			{
			case Type::linear: return "linear";
			case Type::finiteLinear: return "finiteLinear";
			case Type::relu: return "relu";
			case Type::binary: return "binary";
			case Type::gaussian: return "gaussian";
			case Type::sigmoid: return "sigmoid";
			case Type::tanh_: return "tanh";
			}
			return "unknown";
		}
	}
}